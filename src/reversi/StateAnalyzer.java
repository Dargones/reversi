package reversi;

import java.util.*;
import static reversi.BoardState.*;
import static reversi.Disk.*;
import java.util.concurrent.ConcurrentHashMap;
import org.apache.log4j.Logger;

/**
 * Created by alexanderfedchin on 8/31/18.
 * This class represents is a collection of methods for analysis of Board States. Each instance of
 * the class analyzes a specific boardState. It implements runnable because the analysis can
 * (and sometimes should) be concurrent
 */
public class StateAnalyzer implements Runnable {

    private final static Logger logger = Logger.getLogger(StateAnalyzer.class);
    private static final byte TRACE_LEVEL = 13;
    // The level from which to begin to trace analyzed states of the board and print them out.
    // Level is the number of disks already on the board - see BoardState class
    private static final byte MINIMAX_LEVELS_TO_STORE = 9;
    // whenever minimax is used, the program stores minimax value for states that were already seen
    // in a dictionary. However, storing every single seen state would take to much memory, given
    // that this information is only reused during minimax calculation
    private static final byte[] MINIMAX = {0, 0, 0, 0, 13, 0, 0, 0, 0, 0, 0, 0,
            7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0};
    // Levels at which to use minimax and how deep (in levels) the minimax calculations should be.
    private static final byte MULTITHREADING_LEVEL = 22;
    // Level from which to start taking advantage of multithreading
    private static final int LOG_FQ = (int) Math.pow(2, 26);
    // frequency of the report. Report is printed then count % REPORT_FQ == 0
    private static final long DICT_MAX_SIZE = Runtime.getRuntime().maxMemory() / MAX / 16;
    // maximum total number of states that the program will attempt to store in the dictionary.
    // 2 * BoardState.MAX + 10 is the minimum number of bites needed to store an instance of BoardState, 
    // so I assume the actual memory usage to be at most 8 times as much.

    public static ConcurrentHashMap<BoardState, Disk>[] coincDict = new ConcurrentHashMap[MAX];
    // a list of dictionaries to look up states for which the solution is known (coincidences).
    // The maps are concurrent, because they could be accessed simultaneously by multiple threads
    public static byte coincLevel = MAX - 4;
    // The level from which to begin to look up the state inside the coincDict
    // this level can change depending on how much memory the program has
    private static long inTheDict = 0;
    // the number of elements in the coincDictionary. If this value reaches
    // DICT_SIZE_MAX, a whole level is released from the dictionary and the
    // coincLevel is lowered by one
    private static ConcurrentHashMap<BoardState, Byte>[] minimaxDict = new ConcurrentHashMap[MAX];
    // same as coincDict, but for minimax values (which are score estimation, i.e bytes, not Disks)
    // The maps are concurrent, because they could be accessed simultaneously by multiple threads
    private static boolean terminateThreads = false; // True, only if a thread
    // has found a winning solution and all other threads should be terminated
    private static boolean truceSolutionFound = false; // True, only if a thread has found a
    // solution leading to a truce
    private static int reportsPrinted = 0; // Number of reports printed
    // The fields below are needed for weka, but are otherwise irrelevant
    private static Byte lastMinimaxScore = null; // the score predicted by minimax last time
    // it was executed


    static {
        logger.info("Total memory: " + Runtime.getRuntime().maxMemory() +
                        "\nAt most " + DICT_MAX_SIZE + " board-states will be stored"); 
        
        // Modifying the MINIMAX array
        for (byte i = 1; i < MINIMAX.length; i++)
            if ((MINIMAX[i] == 0) && (MINIMAX[i - 1] > 0))
                MINIMAX[i] = (byte) (MINIMAX[i - 1] - 1);
        for (byte i = 0; i < coincDict.length; i++) {
            coincDict[i] = new ConcurrentHashMap<>();
            minimaxDict[i] = new ConcurrentHashMap<>();
        }
    }

    private BoardState state; // the boardState to be analysed
    private Byte minimaxScore = null; // the score assigned to the BoardState by the minimax
    // algorithm. NOTE: Since minimax algorithm is used to predict optimal move,
    // but not to prove what the optimal move is, the minimaxScore can differ from the actual
    // score (which is the goal of this program to calculate). The score is positive, if the current
    // player is expected to win

    /**
     * Default constructor
     * @param state
     */
    public StateAnalyzer(BoardState state) {
        this.state = state;
    }


    /**
     * Traverse the game tree to find a solution
     *
     * @param reverse: 1, if the original player to make a move cannot
     *                 make the move, 2, if both players cannot make a move, 0 - otherwise
     * @return NONE, WHITE or DARK depending on who wins
     */
    public Disk analyze(int reverse) {
        if (terminateThreads) // if one thread has found a move leading to victory,
            // terminate all threads looking at other moves
            return NONE;

        Disk solution = getKnownSolution(reverse); // if a solution is already known, return it
        if (solution != null)
            return solution;

        ArrayList<BoardState> moves = getMoves();
        // If no moves can be made, change the player and call itself
        if ((moves == null) || (moves.size() == 0)) {
            state.reverseTurn();
            return returnResult(analyze(reverse + 1));
        }

        updateLogIfNeeded(moves.size());

        if (state.getLevel() == MULTITHREADING_LEVEL) // use multithreading
            return returnResult(multiThreadedAnalyze(moves));

        int level = state.getLevel();
        // This code is only reached, if level != MULTITHREADING_LEVEL
        boolean trucePossible = false; // whether there is a move that leads to truce
        for (BoardState move : moves) {
            Disk winner = new StateAnalyzer(move).analyze();
            if (level < MULTITHREADING_LEVEL)
                Main.currBF[level - 1] -= 1; // update brancing factors info
            if (winner == state.getTurn())
                return returnResult(state.getTurn());
            if (winner == NONE)
                trucePossible = true;
        }
        if (trucePossible)
            return returnResult(NONE);
        return returnResult(getReverse(state.getTurn()));
    }

    /**
     * If the solution for the current board state is either known or can be easily calculated,
     * because this is the terminal state, then return that solution. Otherwise, return null
     *
     * @param reverse See analyze for this parameter
     * @return
     */
    private Disk getKnownSolution(int reverse) {
        int level = state.getLevel(); // number of disks on the board
        if ((level == MAX) || (reverse == 2)) { // if the game has finished
            Main.levCount[level - 1] += 1; // updating the statistics
            if (state.scores[WHITE.id] > state.scores[DARK.id])
                return WHITE;
            if (state.scores[DARK.id] > state.scores[WHITE.id])
                return DARK;
            return NONE;
        }

        if (level <= coincLevel) {
            // check if the winner for this state was already calculated
            if (level <= TRACE_LEVEL) // for debug purposes only
                logger.info("Currently analyzing:\n" + this.state);
            Disk tmp = coincDict[level - 1].get(state);
            if (tmp != null) // if the value of this state was calculated before
                Main.coincCount[level - 1] += 1;
            return tmp;
        }

        return null;
    }

    /**
     * Update the log if time has come
     *
     * @param movesSize number of moves taht can be made from this state
     */
    private void updateLogIfNeeded(int movesSize) {
        int level = state.getLevel();
        Main.count += 1; // update the total number of states analyzed
        if ((level <= MULTITHREADING_LEVEL) &&
                ((Main.count / LOG_FQ) > reportsPrinted)) { // if it is time to print some output
            reportsPrinted = (int) (Main.count / LOG_FQ);
            Main.updateLog();
        }

        if (level < MULTITHREADING_LEVEL)
            Main.currBF[level - 1] = (long) movesSize; // for logging
    }

    /**
     * Get all possible moves that can be made from this state. If this is a level at which
     * MINIMAX has to be used, use minimax to sort the moves by how useful they are
     * @return
     */
    private ArrayList<BoardState> getMoves() {
        int level = state.getLevel();
        if (MINIMAX[level - 1] != 0) {
            StateClassifier classifier = StateClassifier.getStateClassifier(MINIMAX[level - 1] + level);
            return minimax((byte) (MINIMAX[level - 1] + level),
                    (byte) (level + MINIMAX_LEVELS_TO_STORE), classifier);
        } else
            return state.getMoves(true);
    }

    /**
     *  Launch a thread for each possible move.
     *  The hope is that, if victory is not possible, the threads will concurrently
     *  infer that every move is a loss
     *
     * @param moves all the moves that can be made from this state
     * @return
     */
    private Disk multiThreadedAnalyze(ArrayList<BoardState> moves) {
        Thread[] threads = new Thread[moves.size()];
        for (int i = 0; i < moves.size(); i++) {
            threads[i] = new Thread(new StateAnalyzer(moves.get(i)));
            threads[i].start();
        }
        try {
            for (int i = 0; i < moves.size(); i++)
                threads[i].join();
        } catch (Exception ex) {
            logger.error("An issue with Multithreading. See StateAnalyzer");
            System.exit(-1);
        }
        if (terminateThreads) {// if threads were terminated, there is a thread that found a move
            // that guarantees victory for the player that is to make a move right now
            terminateThreads = false;
            truceSolutionFound = false;
            return state.getTurn();
        }
        if (truceSolutionFound) {
            truceSolutionFound = false;
            return NONE;
        }
        return getReverse(state.getTurn());
    }

    /**
     * Does the necessary postprocessing of the result of the analyze method.
     * Updates statistics, etc.
     *
     * @param result
     * @return
     */
    private Disk returnResult(Disk result) {
        int level = state.getLevel();
        Main.levCount[level - 1] += 1;
        Main.lastTimeUpdated[level - 1] = Main.count;
        if (level <= coincLevel) { // record information about who wins in this state
            coincDict[level - 1].putIfAbsent(state, result);
            // putIfAbsent has to be used due to concurrency issues
            inTheDict += 1;
            if ((level <= MULTITHREADING_LEVEL) && (inTheDict > DICT_MAX_SIZE)) {
                // reduce the size of the dictionary if it is too large
                logger.info("coincLevel reduced to " + (coincLevel - 1));
                inTheDict -= coincDict[coincLevel - 1].size();
                coincDict[coincLevel - 1] = null;
                coincLevel -= 1;
            }
        }
        if (MINIMAX[level - 2] < MINIMAX[level - 1]) {
            // If minimax values are calculated at this level
            int i = level - 1;
            while (MINIMAX[i] != 0) {
                minimaxDict[i] = new ConcurrentHashMap<>();
                i += 1;
            }
        }
        if (level == MULTITHREADING_LEVEL + 1) {
            if (result == getReverse(state.getTurn())) {
                terminateThreads = true; // if one thread has found a move leading to victory,
                // terminate all threads looking at other moves
            } else if (result == NONE) // a solution leading to truce was found
                truceSolutionFound = true;
        }
        return result;
    }

    /**
     * Default version of analyze
     *
     * @return
     */
    public Disk analyze() {
        return analyze((byte) 0);
    }

    /**
     * Default version of minimax. See documentation for the full version of minimax
     * @param maxDepth
     * @param depthInDict
     * @param classifier
     * @return
     */
    private ArrayList<BoardState> minimax(int maxDepth, int depthInDict, StateClassifier classifier) {
        return minimax(maxDepth, depthInDict, classifier, null, true, 0);
    }

    /**
     *
     Traverse the game tree down to the max_depth depth and then return the
     * list of possible moves in the order which guarantees the best score
     * for the player on the max_depth depth.
     * WHITE tries minimizing, DARK tries maximizing
     * @param maxDepth   The depth at which to stop expanding the nodes of the tree, and use
     *                   the classifier if one is given
     * @param depthInDict The last level for which there is a dictionary where all previously seen
     *                    states are recorded
     * @param scoreAbove The score that the parent state currently has. This is needed for
     *                   Alpa-beta pruning to work
     * @param sort  whether to sort the resulting states by how likely they are going to lead the
     *              current player to victory
     * @param reverse See the entry for analyze() for this one
     * @return
     */
    private ArrayList<BoardState> minimax(int maxDepth, int depthInDict, StateClassifier classifier,
                                          Byte scoreAbove, boolean sort, int reverse) {
        //TODO: Make use of coincDict here
        minimaxScore = null;
        int level = state.getLevel();
        if (reverse == 2)
            return null; // no moves are possible from this state

        if (maxDepth == level) { // if this is a leaf, use classifier to get the value at that leaf
            if (classifier != null)
                minimaxScore = classifier.classify(state);
            return null;
        }

        // get all the possible moves that can be reached from this state
        ArrayList<BoardState> moves = state.getMoves(true);
        StateAnalyzer[] analyzers = new StateAnalyzer[moves.size()];
        for (int i = 0; i < moves.size(); i++)
            analyzers[i] = new StateAnalyzer(moves.get(i));

        if (moves.size() == 0) { // if the player to make a move cannot make a move, switch players
            state.reverseTurn();
            minimax(maxDepth, depthInDict, classifier, null, false, reverse + 1);
            state.reverseTurn();
            return null;
        }

        for (int i = 0; i < moves.size(); i++) { // i is also used to index anayzers
            BoardState move = moves.get(i);
            int currScore;
            Byte dictEntry = null;
            // see if the score for this state was already precalculated
            if (level <= depthInDict)
                dictEntry = minimaxDict[level - 2].get(move);
            if (dictEntry != null)
                analyzers[i].minimaxScore = minimaxDict[level - 2].get(move);
            else {
                analyzers[i].minimax(maxDepth, depthInDict, classifier, minimaxScore, false, 0);
                if (level <= depthInDict)
                    minimaxDict[level - 2].putIfAbsent(move, analyzers[i].getScore());
            }
            currScore = -analyzers[i].getScore();

            if ((minimaxScore == null) || (currScore > minimaxScore))
                minimaxScore = (byte) currScore; // update the score, if a new best move is found

            // Alpha-beta pruning:
            if ((scoreAbove != null) && (-minimaxScore <= scoreAbove))
                return null;
        }

        if (sort) {
            Arrays.sort(analyzers, Comparator.comparingInt(StateAnalyzer::getScore));
            moves = new ArrayList<>();
            for (StateAnalyzer analyzer: analyzers)
                moves.add(analyzer.state);
            lastMinimaxScore = (byte) (-analyzers[0].getScore());
        }

        return moves;
    }

    /**
     * Get the score assigned to the state by the minimax algorithm. If minimaxScore field is null,
     * return the simple current score difference (how much more of the disks of one color there are
     * on the board than of the disks of the other color)
     * @return
     */
    private byte getScore() {
        if (minimaxScore == null)
            return (byte) (state.getScoreDifference());
        return minimaxScore;
    }

    /**
     * Randomly sample COUNT states from level LEVEL and use minimax to get the score
     * originating from these states. If necessary, use CLASSIFIER at the leafs of minimax.
     * In essence, this method creates a dataset for a classifier to be trained on
     * @param level  level at which to take the states
     * @param count  number of states to return
     * @param evaluationLevel level at which to evaluate the leafs in minimax
     * @param classifier classifier to use to evaluate the leaves
     * @return
     */
    public static DataSet createDataset(int level, int count,
                                                 int evaluationLevel, StateClassifier classifier) {
        BoardState[] features = new BoardState[count];
        Byte[] labels = new Byte[count];
        Random random = new Random();
        logger.info("Creating dataset for level:" + level + " ev_level:" + evaluationLevel);
        long timeStart = System.currentTimeMillis();
        int i = 0;
        while (i < count) {
            if ((i % 100 == 0)) {
                long timeLeft = (long) ((double) (System.currentTimeMillis() - timeStart) /
                        i * (count - i));
                logger.info("Time left:" + Main.getDuration(timeLeft));
            }
            BoardState currState = new BoardState();
            int currLevel = currState.getLevel();
            boolean turnFlipped = false;
            while (currLevel < level) {
                ArrayList<BoardState> moves = currState.getMoves(false);
                if (moves.size() == 0) {
                    if (turnFlipped)
                        break;
                    else {
                        turnFlipped = true;
                        currState.reverseTurn();
                        continue;
                    }
                } else
                    turnFlipped = false;
                currState = moves.get(random.nextInt(moves.size()));
                currLevel++;
            }
            if (level != currLevel)
                continue;
            if (coincDict[currLevel - 1].get(currState) != null)
                continue;
            StateAnalyzer analyzer = new StateAnalyzer(currState);
            analyzer.minimaxScore = null;

            analyzer.minimax(evaluationLevel, level + MINIMAX_LEVELS_TO_STORE, classifier);
            for (byte j = 0; j < coincDict.length; j++) // clearing minimax dictionaries
                minimaxDict[j] = new ConcurrentHashMap<>();
            features[i] = currState;
            labels[i] = lastMinimaxScore;
            i++;
        }
        return new DataSet(features, labels);
    }

    @Override
    public void run() {
        analyze();
    }

    /**
     * Eager to replace this with Kotlin "data class":)
     */
    static class DataSet {
        BoardState[] instances;
        Byte[] labels;

        DataSet (BoardState[] instances, Byte[] labels) {
            this.instances = instances;
            this.labels = labels;
        }
    }
}