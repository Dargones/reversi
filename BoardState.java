package reversi;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Created by alexanderfedchin on 8/31/18.
 * This class represents a BoardState and contains multiple tool for the analysis of the state
 */
public class BoardState implements Runnable{

    private static final byte MIMO = Main.MAX_IND - 1;
    // Max Ind Minus One - to speed things up
    private static final byte[][] DIRS = {{0, 1}, {1, 1}, {1, 0}, {1, -1},
            {0, -1}, {-1, -1}, {-1, 0}, {-1, 1}};
    // the 8 directions in which one can go from a tile.
    private static final byte TRACE_LEVEL = 13;
    // The level from which to begin to trace the states of the board and print them.
    // Level is the number of disks already on the board
    private static final byte MINIMAX_LEVELS_TO_STORE = 9;
    // whenever minimax is used, the program stores minimax value for states that were already seen
    // in a dictionary. However, storing every single seen state would take to much memory.
    // For this reason, only the first MINIMAX_LEELS_TO_STORE levels from the initial state are
    // being kept in the dictionary
    private static final byte[] MINIMAX = {0, 0, 0, 0, 13, 0, 0, 0, 0, 0, 0, 0,
            7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0};
    // Levels at which to use minimax and how deep (in levels) the minimax calculations should be.
    private static final byte MULTIPROCESSING_LEVEL = 22;
    // Level from which to start taking advantage of multiprocessing
    // IMPORTANT: Multiprocessing level should be below the first MINIMAX level
    private static final int REPORT_FQ = (int) Math.pow(2, 23);
    // frequency of the report. Report is printed then count % REPORT_FQ == 0
    private static final Transformation[] TRANSFORMS = {
            (x, y) -> new byte[]{(byte) (MIMO - x), y}, (x, y) -> new byte[]{y, x},
            (x, y) -> new byte[]{x, (byte) (MIMO - y)},
            (x, y) -> new byte[]{(byte) (MIMO - y), (byte) (MIMO - x)},
            (x, y) -> new byte[]{x, y},
            (x, y) -> new byte[]{(byte) (MIMO - x), (byte) (MIMO - y)},
            (x, y) -> new byte[]{y, (byte) (MIMO - x)},
            (x, y) -> new byte[]{(byte) (MIMO - y), x}};
    // These lambda functions represent the four rotations and the four
    // reflections which can be applied to a BoardState. The functions are only
    // used inside the static block and during the debug process.
    private static byte[][][] ITERATORS;
    // 8 lists of coordinates, which
    // correspond to 8 transformations. E.g. ITERATORS[2][i] stores the
    // coordinates which one gets by applying TRANSFORMS[2] to ITERATORS[0][i]
    private static final long DICT_MAX_SIZE = 10000000;

    public static byte coincLevel = Main.MAX - 4;
    // The level from which to begin to look up the state inside the coincDict
    // this level can change depending on how much memory the program has
    public static FourLayeredHashMap[] coincDict = new FourLayeredHashMap[Main.MAX];
    // a list of dictionaries to look up states for which the solution is known
    private static long inTheDict = 0;
    // the number of elements in the coincDictionary. If this value reaches
    // DICT_SIZE_MAX, a whole level is released from the dictionary and the
    // coincLevel is lowered by one
    private static FourLayeredHashMap[] minimaxDict = new FourLayeredHashMap[Main.MAX];
    // same for minimax. The difference is in that in the minimax dictionary the
    // actual minimax values are stored
    private static boolean terminateThreads = false; // True, only if a thread
    // has found a winning solution and all other threads should be terminated
    private static int reportsPrinted = 0; // Number of reports printed
    // The fields below are needed for weka, but are otherwise irrelevant
    public static FastVector attributes = new FastVector();
    private static final int MAX_MINIMAX_INSTANCES = 10000;
    private static Instances currMinimax = new Instances("currMinimax", attributes,
            MAX_MINIMAX_INSTANCES);


    static {
        // Modifying the MINIMAX array
        for (byte i = 1; i < MINIMAX.length; i++)
            if ((MINIMAX[i] == 0) && (MINIMAX[i - 1] > 0))
                MINIMAX[i] = (byte) (MINIMAX[i - 1] - 1);
        for (byte i = 0; i < coincDict.length; i++) {
            coincDict[i] = new FourLayeredHashMap();
            minimaxDict[i] = new FourLayeredHashMap();
        }

        // Building the ITERATORS Arrays using TRANSFORMS
        ITERATORS = new byte[TRANSFORMS.length][][];
        for (byte k = 0; k < TRANSFORMS.length; k++) {
            ITERATORS[k] = new byte[Main.MAX][];
            for (byte i = 0; i < Main.MAX_IND; i++)
                for (byte j = 0; j < Main.MAX_IND; j++)
                    ITERATORS[k][i * Main.MAX_IND + j] = TRANSFORMS[k].transform(i, j);
        }

        // The code below is needed for weka, but is otherwise irrelevant

        attributes.addElement(new Attribute("player_1_score", 0));
        attributes.addElement(new Attribute("player_-1_score", 1));
        attributes.addElement(new Attribute("player_1_fixed", 2));
        attributes.addElement(new Attribute("player_-1_fixed", 3));

        for (byte i = 0; i < Main.MAX_IND; i++)
            for (byte j = 0; j < Main.MAX_IND; j++)
                attributes.addElement(new Attribute(i + " " + j, 4 + i * Main.MAX_IND + j));

        for (byte i = 0; i < Main.MAX_IND; i++)
            for (byte j = 0; j < Main.MAX_IND; j++)
                attributes.addElement(new Attribute("f " + i + " " + j, 4 + Main.MAX + i * Main.MAX_IND + j));

        Attribute label = new Attribute("label", 4 + 2 * Main.MAX);
        attributes.addElement(label);
        currMinimax.setClassIndex(attributes.size() - 1);
    }

    private byte[][] board, fixed;
    // board is a two dimensional array where each element is either Main.DARK, Main.WHITE, or
    // Main.TRUCE (for unoccupied tiles) and represents the current state of the Board.
    // fixed - is a matrix similar to board, except taht it only stores information about disks that
    // cannot be flipped
    private byte[] hor, ver, diaS, diaI, scores, fixedScores;
    // scores and fixedScores are two arrays that keep track of the number of all (scores) and
    // unflippable (fixedScores) white and dark disks on the board resp. hor, ver, diaS, and diaI
    // keep track of the number of disks on particular horizontal lines, vertical lines, etc.
    private byte turn; // whose turn is next. Either Main.DARK or Main.WHITE
    private Byte minimaxScore; // the score assigned to the BoardState by the minimax algorithm
    // NOTE: Since minimax algorithm is used to predict optimal move, but not to prove what the
    // optimal move is, the minimaxScore can differ from the actual score (which is the goal of
    // this prograp to calculate)

    /**
     * Default constructor
     */
    public BoardState() {
        // creating an empty board
        board = new byte[Main.MAX_IND][Main.MAX_IND];
        for (byte i = 0; i < board.length; i++)
            Arrays.fill(board[i], Main.TRUCE);
        // the board with fixed corners marked
        fixed = new byte[Main.MAX_IND][Main.MAX_IND];
        for (byte i = 0; i < fixed.length; i++)
            Arrays.fill(fixed[i], Main.TRUCE);
        scores = new byte[]{2, 2};
        fixedScores = new byte[]{0, 0};
        // putting a square in the middle of the board
        int b = Main.MAX_IND / 2;
        int a = b - 1;
        board[a][a] = Main.WHITE;
        board[a][b] = Main.DARK;
        board[b][b] = Main.WHITE;
        board[b][a] = Main.DARK;
        turn = Main.DARK;
        hor = new byte[Main.MAX_IND];
        Arrays.fill(hor, Main.MAX_IND);
        // for each horizontal line, how many empty spots are left
        ver = Arrays.copyOf(hor, Main.MAX_IND);
        // same for each vertical line
        diaS = new byte[Main.MAX_IND + MIMO];
        for (byte i = 0; i < Main.MAX_IND; i++) {
            diaS[i] = (byte) (i + 1);
            diaS[MIMO - i] = (byte) (i + 1);
        }
        diaI = Arrays.copyOf(ver, Main.MAX_IND + MIMO);
        // same for the two types of diagonal lines
        int[][] pos = {{a, a}, {a, b}, {b, b}, {b, a}};
        for (byte i = 0; i < pos.length; i++) {
            hor[pos[i][0]] -= 1;
            ver[pos[i][1]] -= 1;
            diaI[MIMO - pos[i][1] + pos[i][0]] -= 1;
            diaS[pos[i][0] + pos[i][1]] -= 1;
        }
        minimaxScore = null;
    }

    /**
     * Create a copy of an original
     * @param original
     */
    public BoardState(BoardState original) {
        board = new byte[Main.MAX_IND][];
        for (byte i = 0; i < board.length; i++)
            board[i] = Arrays.copyOf(original.board[i], Main.MAX_IND);
        fixed = new byte[Main.MAX_IND][];
        for (byte i = 0; i < fixed.length; i++)
            fixed[i] = Arrays.copyOf(original.fixed[i], Main.MAX_IND);
        hor = Arrays.copyOf(original.hor, Main.MAX_IND);
        ver = Arrays.copyOf(original.ver, Main.MAX_IND);
        diaS = Arrays.copyOf(original.diaS, Main.MAX_IND + MIMO);
        diaI = Arrays.copyOf(original.diaI, Main.MAX_IND + MIMO);
        scores = Arrays.copyOf(original.scores, 2);
        fixedScores = Arrays.copyOf(original.fixedScores, 2);
        turn = original.turn;
        minimaxScore = null;
    }

    /**
     * Traverse the game tree to find a solution
     *
     * @param reverse: 1, if the original player to make a move cannot
     *                 make the move, 2, if both players cannot make a move, 0 - otherwise
     * @return TRUCE, WHITE or DARK depending on who wins
     */
    public byte analyze(int reverse) {
        if (terminateThreads) // if one thread has found a move leading to victory,
            // terminate all threads looking at other moves
            return Main.TRUCE;

        int level = scores[Main.WHITE] + scores[Main.DARK]; // number of disks on the board
        if ((level == Main.MAX) || (reverse == 2) ||
                (fixedScores[Main.WHITE] > Main.MAX / 2) ||
                (fixedScores[Main.DARK] > Main.MAX / 2)) { // if the winner is apparent
            if (level == Main.MAX)
                assert Main.sum(fixedScores) == Main.MAX;
            Main.levCount[level - 1] += 1;
            // can be deleted if True
            if (fixedScores[Main.WHITE] > fixedScores[Main.DARK])
                return Main.WHITE;
            if (fixedScores[Main.DARK] > fixedScores[Main.WHITE])
                return Main.DARK;
            return Main.TRUCE;
        }

        int[] curr = null; // the code of the current state analyzed
        if (level <= coincLevel) {
            // check if the winner for this state was already calculated
            curr = getCode();
            if (level <= TRACE_LEVEL) { // for debug purposes only
                System.out.println(this);
            }
            Byte tmp = coincDict[level - 1].get(curr);
            if (tmp != null) { // if the value of this state was calculated before
                Main.coincCount[level - 1] += 1;
                return tmp;
            }
        }

        Main.count += 1; // total number of states analyzed
        if ((level <= MULTIPROCESSING_LEVEL) &&
                ((Main.count / REPORT_FQ) > reportsPrinted)) { // if it is time to print some output
            reportsPrinted = (int) (Main.count / REPORT_FQ);
            Main.printReport();
        }

        // Get all possible moves that can be made from this state. If this is a level at which
        // MINIMAX has to be used, use minimax to sort the moves by how useful they are
        ArrayList<BoardState> moves;
        if (MINIMAX[level - 1] != 0) {
            Classifier classifier = Predictor.getClassifier(MINIMAX[level - 1] + level);
            moves = minimax((byte) (MINIMAX[level - 1] + level), null, (byte) 0, true,
                    (byte) (level + MINIMAX_LEVELS_TO_STORE), classifier);
        } else
            moves = getMoves(true);

        // If no moves can be made, change the player and call itself
        if ((moves == null) || (moves.size() == 0)) {
            turn = (byte) (1 - turn);
            return returnResult(curr, level, analyze(reverse + 1));
        }

        if (level == MULTIPROCESSING_LEVEL) {
            // if multiprocessing is used at this level, launch a thread for each possible move.
            // The hope is that, if victory is not possible, the threads will concurrently
            // infer that every move is a loss
            Thread[] threads = new Thread[moves.size()];
            for (int i = 0; i < moves.size(); i++) {
                threads[i] = new Thread(moves.get(i));
                moves.get(i).minimaxScore = null;
                threads[i].start();
            }
            try {
                for (int i = 0; i < moves.size(); i++)
                    threads[i].join();
            } catch (Exception ex) {
                System.out.println("problem");
            }
            if (terminateThreads) {
                terminateThreads = false;
                return returnResult(curr, level, turn);
            }
            for (int i = 0; i < moves.size(); i++) {
                if (moves.get(i).minimaxScore == Main.TRUCE)
                    return returnResult(curr, level, Main.TRUCE);
            }
            return returnResult(curr, level, (byte) (1 - turn));

        } else if (level < MULTIPROCESSING_LEVEL)
            Main.currBF[level - 1] = (long) moves.size(); // for logging

        // This code is only reached, if level != MULTIPROCESSING_LEVEL
        boolean trucePossible = false; // whether there is a move that leads to truce
        for (BoardState move : moves) {
            byte winner = move.analyze();
            if (level < MULTIPROCESSING_LEVEL)
                Main.currBF[level - 1] -= 1;
            if (winner == turn)
                return returnResult(curr, level, turn);
            if (winner == Main.TRUCE)
                trucePossible = true;
        }
        if (trucePossible)
            return returnResult(curr, level, Main.TRUCE);
        return returnResult(curr, level, (byte) (1 - turn));
    }

    /**
     * Default version of analyze
     *
     * @return
     */
    public byte analyze() {
        return analyze((byte) 0);
    }

    /**
     * Randomly sample states COUNT states from level LEVEL and use minimax to get the score
     * originating from these states. If necessary, use CLASSIFIER at the leafs of minimax.
     * In essence, this method creates a dataset for a classifier to be trained on
     * @param level  level at which to take the states
     * @param count  number of states to return
     * @param name   name of the attribute
     * @return
     */
    public Instances getInstances(byte level, int count, String name,
                                  int evaluationLevel, Classifier classifier) {
        Instances instances = new Instances(name, attributes, count);
        instances.setClassIndex(attributes.size() - 1);
        Random random = new Random();
        int i = 0;
        while (i < count) {
            if (i % 100 == 0) {
                //System.out.println((count - i) + " states left");
                long timeLeft = (long) ((double) (System.currentTimeMillis() - Predictor.timeStart) / i * (count - i));
                System.out.println("Time left:" + Main.getDuration(timeLeft));
            }
            int currLevel = scores[Main.WHITE] + scores[Main.DARK];
            BoardState currState = this;
            boolean turnFlipped = false;
            while (currLevel < level) {
                ArrayList<BoardState> moves = currState.getMoves(false);
                if (moves.size() == 0) {
                    if (turnFlipped)
                        break;
                    else {
                        turnFlipped = true;
                        currState.turn = (byte) (1 - currState.turn);
                        continue;
                    }
                } else
                    turnFlipped = false;
                currState = moves.get(random.nextInt(moves.size()));
                currLevel = currState.scores[Main.WHITE] +
                        currState.scores[Main.DARK];
            }
            if (level != currLevel)
                continue;
            if (coincDict[currLevel - 1].get(currState.getCode()) != null)
                continue;
            Instance instance = currState.stateToInstance(true, evaluationLevel, classifier);
            coincDict[currLevel - 1].putIfAbsent(currState.getCode(), (byte) instance.value(attributes.size() - 1));
            instances.add(instance);
            i++;
        }
        return instances;
    }

    /**
     * Convert the state to a weka Instance that can be used for training or
     * evaluation purposes
     * @param assessLabel  if True, calculate the winner for the state
     * @return
     */
    public Instance stateToInstance(boolean assessLabel, int evaluationLevel, Classifier classifier) {
        Instance instance = new weka.core.Instance(attributes.size());
        instance.setValue((Attribute) attributes.elementAt(0), scores[turn]);
        instance.setValue((Attribute) attributes.elementAt(1), scores[1 - turn]);
        instance.setValue((Attribute) attributes.elementAt(2), fixedScores[turn]);
        instance.setValue((Attribute) attributes.elementAt(3), fixedScores[1 - turn]);

        for (byte i = 0; i < Main.MAX_IND; i++)
            for (byte j = 0; j < Main.MAX_IND; j++) {
                int value = (board[i][j] == turn) ? 1 : ((board[i][j] == 1 - turn) ? -1 : 0);
                instance.setValue((Attribute) attributes.elementAt(4 + i * Main.MAX_IND + j), value);
            }

        for (byte i = 0; i < Main.MAX_IND; i++)
            for (byte j = 0; j < Main.MAX_IND; j++) {
                int value = (fixed[i][j] == turn) ? 1 : ((fixed[i][j] == 1 - turn) ? -1 : 0);
                instance.setValue((Attribute) attributes.elementAt(4 + Main.MAX + i * Main.MAX_IND + j), value);
            }

        if (assessLabel) { // if the level should be assessed (this is only done to train the classifier)
            ArrayList<BoardState> moves = minimax((byte) (evaluationLevel), null, (byte) 0, true,
                        (byte) (evaluationLevel - 1), classifier);
            if (moves == null) {
                turn = (byte) (1 - turn);
                moves = minimax((byte) (evaluationLevel), null, (byte) 0, true,
                        (byte) evaluationLevel, classifier);
            }
            int value;
            if (moves == null)
                value = scores[turn] - scores[1 - turn];
            else if (turn == Main.DARK)
                value = moves.get(0).getScore();
            else
                value = -moves.get(0).getScore();

            for (int i = scores[Main.WHITE] + scores[Main.DARK] - 1; i < Main.MAX; i++)
                minimaxDict[i] = new FourLayeredHashMap();

            instance.setValue((Attribute) attributes.elementAt(attributes.size() - 1), value);
        }

        return instance;
    }

    /**
     * Does the necessary postprocessing of the result of the analyze method.
     * Updates statistics, etc.
     *
     * @param curr
     * @param level
     * @param result
     * @return
     */
    private byte returnResult(int[] curr, int level, byte result) {
        Main.levCount[level - 1] += 1;
        Main.lastTimeUpdated[level - 1] = Main.count;
        if (level <= coincLevel) { // record information about who wins in this state
            coincDict[level - 1].putIfAbsent(curr, result);
            inTheDict += 1;
            if ((level <= MULTIPROCESSING_LEVEL) && (inTheDict > DICT_MAX_SIZE)) {
                // reduce the size of the dictionary if it is too large
                System.out.println("coincLevel reduced to " + (coincLevel - 1));
                inTheDict -= coincDict[coincLevel - 1].size();
                coincDict[coincLevel - 1] = null;
                coincLevel -= 1;
            }
        }
        if (MINIMAX[level - 2] < MINIMAX[level - 1]) {
            // If minimax values are calculated at this level
            int i = level - 1;
            while (MINIMAX[i] != 0) {
                //TODO: What should be the initial capacity?
                minimaxDict[i] = new FourLayeredHashMap();
                i += 1;
            }
        }
        if ((level == MULTIPROCESSING_LEVEL + 1) && (result == 1 - turn))
            terminateThreads = true; // if one thread has found a move leading to victory,
            // terminate all threads looking at other moves
        return result;
    }


    /**
     *
     Traverse the game tree down to the max_depth depth and then return the
     * list of possible moves in the order which guaranteers the best score
     * for the player on the max_depth depth.
     * WHITE tries minimizing, DARK tries maximizing
     * @param maxDepth   The depth at which to stop expanding the nodes of the tree, and use
     *                   the classifier if one is given
     * @param scoreAbove The score that the parent state currently has. This is needed for
     *                  Alpa-beta pruning to work
     * @param reverse See the entry for analyze() for this one
     * @return
     */
    private ArrayList<BoardState> minimax(byte maxDepth, Byte scoreAbove,
                                          byte reverse, boolean sort, byte depthInDict, Classifier classifier) {
        //TODO make use of coincDict here?
        minimaxScore = null;
        int level = scores[Main.WHITE] + scores[Main.DARK];
        if ((fixedScores[Main.WHITE] > Main.MAX / 2) ||
                (fixedScores[Main.DARK] > Main.MAX / 2) || (reverse == 2))
            return null;

        if (maxDepth == level) { // if this is a leaf, use classifier to get the value at that leaf
            if (classifier != null) {
                if (currMinimax.numInstances() == MAX_MINIMAX_INSTANCES) {
                    currMinimax = new Instances("currMinimax", attributes, MAX_MINIMAX_INSTANCES);
                    currMinimax.setClassIndex(attributes.size() - 1);
                }
                Instance wekaRepresentation = stateToInstance(false, -1, null);
                currMinimax.add(wekaRepresentation);
                try {
                    minimaxScore = (byte) Math.round(classifier.classifyInstance(wekaRepresentation));
                    if (turn == Main.WHITE)
                        minimaxScore = (byte) -minimaxScore;
                } catch (Exception ex) {
                    System.out.print("Problem here");
                }
            }
            return null;
        }

        // get all the possible moves taht can be made from this state
        ArrayList<BoardState> moves = getMoves(false);

        if (moves.size() == 0) { // if the player to make a ove cannot make a move, switch players
            turn = (byte) (1 - turn);
            minimax(maxDepth, null, (byte) (reverse + 1), false, depthInDict, classifier);
            turn = (byte) (1 - turn);
            return null;
        }

        for (BoardState move:moves) {
            byte currScore;
            int[] tmp = move.getCode();
            Byte dictEntry = null;
            // see if the score for this state was already precalculated
            if (level <= depthInDict)
                dictEntry = minimaxDict[level - 2].get(tmp);
            if (dictEntry != null) {
                currScore = minimaxDict[level - 2].get(tmp);
                move.minimaxScore = currScore;
            } else {
                move.minimax(maxDepth, minimaxScore, (byte) 0, false, depthInDict, classifier);
                currScore = move.getScore();
                if (level <= depthInDict)
                    minimaxDict[level - 2].putIfAbsent(tmp, currScore);
            }

            if ((minimaxScore == null) || ((turn == Main.WHITE) &&
                    (currScore < minimaxScore)) || ((turn == Main.DARK) &&
                    (currScore > minimaxScore)))
                minimaxScore = currScore; // update the score, if a new best move is found

            // Alpha-beta pruning:
            if ((turn == Main.WHITE) && (scoreAbove != null) && (minimaxScore <= scoreAbove))
                return null;
            if ((turn == Main.DARK) && (scoreAbove != null) && (minimaxScore >= scoreAbove))
                return null;
        }

        /* if (true)
            for (BoardState move:moves)
                System.out.println(move + " " + move.getScore());*/

        if (sort) {
            if (turn == Main.WHITE)
                moves.sort((x, y) -> x.getScore() - y.getScore());
            else
                moves.sort((x, y) -> y.getScore() - x.getScore());
        }
        return moves;
    }

    /**
     * Get the list of possible moves (boards)
     *
     * @param sort if True, sort the moves by the fixed score
     * @return
     */
    private ArrayList<BoardState> getMoves(boolean sort) {
        //TODO: initial capacity
        ArrayList<BoardState> moves = new ArrayList<>();
        for (byte i = 0; i < board.length; i++)
            for (byte j = 0; j < board.length; j++) {
                if (board[i][j] != Main.TRUCE)
                    continue;
                // the tile is not occupied but it is yet to be discovered,
                // whether something can be placed here
                BoardState trial = tryMove(i, j);
                if (trial != null)
                    moves.add(trial);
            }
        if (sort)
            moves.sort((x, y) -> y.fixedScores[turn] - y.fixedScores[1 - turn] -
            x.fixedScores[turn] + x.fixedScores[1 - turn]);
        return moves;
    }

    /**
     * Attempt to place a disk at (row, column). If this is possible, create a
     * new object describing the changes made to the board
     *
     * @param r the row where the disk is to be placed
     * @param c the column where the disk is to be placed
     * @return
     */
    private BoardState tryMove(byte r, byte c) {
        BoardState result = null;
        LinkedList<byte[]> cellsToCheck = new LinkedList<>();
        // the cells from which checkForCorners method should be ran in the end
        for (byte[] dir : DIRS) {
            boolean dirIsValid = false;
            // there are disks to flip in this direction
            boolean enemySeen = false; // an enemy disk is in this direction
            byte m = 1; // multiplier
            while ((r + dir[0] * m < Main.MAX_IND) && (r + dir[0] * m >= 0) &&
                    (c + dir[1] * m < Main.MAX_IND) && (c + dir[1] * m >= 0)) {
                if (board[r + dir[0] * m][c + dir[1] * m] == turn) {
                    dirIsValid = true;
                    break;
                } else if (board[r + dir[0] * m][c + dir[1] * m] == Main.TRUCE)
                    break;
                else
                    enemySeen = true;
                m += 1;
            }
            if ((!dirIsValid) || (!enemySeen))
                continue;

            if (result == null) {
                result = new BoardState(this);
                result.board[r][c] = turn;
                result.scores[turn] += 1;
                result.turn = (byte) (1 - turn);
                result.ver[c] -= 1;
                result.hor[r] -= 1;
                result.diaS[c + r] -= 1;
                result.diaI[MIMO - c + r] -= 1;
                cellsToCheck.add(new byte[]{r, c});
            }

            m = 1;
            while (board[r + dir[0] * m][c + dir[1] * m] != turn) {
                result.board[r + dir[0] * m][c + dir[1] * m] = turn;
                result.scores[turn] += 1;
                result.scores[1 - turn] -= 1;
                m += 1;
            }
            cellsToCheck.add(new byte[]{(byte) (r + dir[0] * (m - 1)),
                    (byte) (c + dir[1] * (m - 1))});
        }
        for (byte[] cell : cellsToCheck)
            result.checkForCorners(cell[0], cell[1]);
        return result;
    }

    /**
     * Get a string representation of the state
     * X corresponds to dark disks
     * 0 corresponds to white disks
     * . corresponds to unoccupied tiles
     *
     * @return
     */
    public String toString(Transformation transform) {
        String result = "";
        byte[] coord;
        for (byte i = 0; i < board.length; i++) {
            for (byte j = 0; j < board[i].length; j++) {
                coord = transform.transform(i, j);
                if (board[coord[0]][coord[1]] == Main.DARK)
                    result += "X";
                else if (board[coord[0]][coord[1]] == Main.WHITE)
                    result += "0";
                else
                    result += ".";
            }
            result += "\t";
            for (byte j = 0; j < fixed[i].length; j++) {
                coord = transform.transform(i, j);
                if (fixed[coord[0]][coord[1]] == Main.DARK)
                    result += "X";
                else if (fixed[coord[0]][coord[1]] == Main.WHITE)
                    result += "0";
                else
                    result += ".";
            }
            result += "\n";
        }
        return result + Arrays.toString((getCode())) + "\n";
    }


    public String toString() {
        return toString((x, y) -> new byte[] {x, y});
    }

    /**
     * Starting from cell (r, c), check if the disk at this position can be flipped or not. If not,
     * update the fixed Array
     * @param r row
     * @param c column
     */
    private void checkForCorners(byte r, byte c) {
        if ((r >= Main.MAX_IND) || (r < 0) || (c >= Main.MAX_IND) || (c < 0))
            // out of bounds
            return;

        if ((fixed[r][c] != Main.TRUCE) || (board[r][c] == Main.TRUCE))
            // this cell is already fixed or should not be fixed
            return;

        if ((c == 0) || (c == MIMO)) {
            if ((r != 0) && (r != MIMO) && (ver[c] != 0) && !checkDirection(
                    (byte) (r + 1), c, (byte) (r - 1), c, board[r][c]))
                return;
        } else if ((hor[r] != 0) && !checkDirection(
                r, (byte) (c + 1), r, (byte) (c - 1), board[r][c]))
            return;
        else if ((r != 0) && (r != MIMO) && !((checkDirection(
                (byte) (r + 1), (byte) (c + 1), (byte) (r - 1), (byte) (c - 1),
                board[r][c]) || diaI[MIMO - c + r] == 0) && (checkDirection(
                (byte) (r - 1), (byte) (c + 1), (byte) (r + 1),
                (byte) (c - 1), board[r][c]) || diaS[c + r] == 0) && (
                checkDirection((byte) (r + 1), c, (byte) (r - 1), c,
                        board[r][c]) || ver[c] == 0)))
            return;

        fixed[r][c] = board[r][c];
        fixedScores[board[r][c]] += 1;
        for (byte[] dir : DIRS)
            checkForCorners((byte) (r + dir[0]), (byte) (c + dir[1]));
    }

    /**
     * Check whether the value of a disk between (r1, c1) and (r2, c2) can be
     * be changed by placing a disk somewhere along (r1, c1) <-> (r2, c2)
     *
     * @param r1
     * @param c1
     * @param r2
     * @param c2
     * @param color color of the disk in question
     * @return
     */
    private boolean checkDirection(byte r1, byte c1, byte r2, byte c2,
                                   byte color) {
        // System.out.println(r1 + " " + r2 + " " + c1 + " " + c2 + " " + color);
        return ((fixed[r1][c1] == color) || fixed[r2][c2] == color || (
                fixed[r1][c1] == (1 - color) && fixed[r2][c2] == (1 - color)));
    }

    /**
     * Get the "code" of the board. The code is such that each reflection or
     * rotation of the same board (but not any other board) has the same code.
     * Note that boards that are one "color flip" from each other will yield
     * different codes.
     * The function loops through all the elements of the board in a specific
     * order and chooses a transformation (a rotation or a reflection) that
     * would correspond to the greatest code. Each two bits in the code are 00,
     * 01, or 10 depending on what kind of disk (dark, white, or empty) stays
     * at which position.
     *
     * @return the code (a long, maybe a 128bit integer)
     */
    private int[] getCode() {
        ArrayList<Byte> its = new ArrayList<>();
        ArrayList<Byte> old_its;
        for (byte i = 0; i < ITERATORS.length; i++)
            its.add(i);
        int[] code = new int[4];
        for (int i = 0; i < Main.MAX; i++) {
            byte max = -1;
            old_its = its;
            for (byte j = 0; j < old_its.size(); j++) {
                byte[] coords = ITERATORS[old_its.get(j)][i];
                if (board[coords[0]][coords[1]] > max) {
                    max = board[coords[0]][coords[1]];
                    its = new ArrayList<>(1);
                }
                if (board[coords[0]][coords[1]] == max)
                    its.add(old_its.get(j));
            }
            code[i % 4] = (code[i % 4] + max) * 3;
        }
        code[code.length - 1] += turn;
        return code;
    }

    private byte getScore() {
        if (minimaxScore == null)
            return (byte) (fixedScores[Main.DARK] - fixedScores[Main.WHITE]);
        return minimaxScore;
    }

    @Override
    public void run() {
        minimaxScore = analyze();
    }

    /**
     * An interface for TRANSFORM lambda functions at the top
     */
    private interface Transformation {
        byte[] transform(byte x, byte y);
    }
}
