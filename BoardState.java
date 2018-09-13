package reversi;

import java.util.*;

/**
 * Created by alexanderfedchin on 8/31/18.
 */
public class BoardState {

    private static final byte MIMO = Main.MAX_IND - 1;
    // Max Ind Minus One - to speed things up
    private static final byte[][] DIRS = {{0, 1}, {1, 1}, {1, 0}, {1, -1},
            {0, -1}, {-1, -1}, {-1, 0}, {-1, 1}};
    // the 8 directions in which one can go from a tile.
    private static final byte TRACE_LEVEL = 15; //The level from which to begin
    // to trace the states of the board and print them. Level is the number of
    // disks already on the board
    private static final byte MINIMAX_LEVELS_TO_STORE = 6;
    // number of minimax level calculations that should be kept intact (these
    // will not be recalculated but will take up space)
    private static final byte[] MINIMAX = {0, 0, 0, 14, 0, 0, 0, 0, 0, 14, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0};
    // Levels at which to use minimax and how deep the minimax calculations
    // should dive. Note that the actual calculations are made at the levels at
    // which the value of the array increases
    private static final byte MULTIPROCESSING_LEVEL = 40;
    // IMPORTANT: Multiprocessing level should be below the first MINIMAX level
    private static final int REPORT_FQ = (int) Math.pow(2, 18);
    // frequency of the report. Report is printed then count % REPORT_FQ == 0
    private static final Transformation[] TRANSFORMS = {
            (x, y) -> new byte[]{(byte) (MIMO - x), y}, (x, y) -> new byte[]{y, x},
            (x, y) -> new byte[]{x, (byte) (MIMO - y)},
            (x, y) -> new byte[]{(byte) (MIMO - y), (byte) (MIMO - x)},
            (x, y) -> new byte[]{x, y},
            (x, y) -> new byte[]{(byte) (MIMO - x), (byte) (MIMO - y)},
            (x, y) -> new byte[]{y, (byte) (MIMO - x)},
            (x, y) -> new byte[]{(byte) (MIMO - y), x}};
    private static byte[][][] ITERATORS;
    private static long DICT_MAX_SIZE = 3000000;

    private static byte coincLevel = Main.MAX - 2;
    // The level from which to begin to look up the state inside the coincDict
    // this level can change depending on how much memory the program has
    private static long count = 0; // number of different states considered while
    // traversing the game. The count does not include the final states
    private static HashMap<StateCode, Byte>[] coincDict = new HashMap[Main.MAX];
    // a list of dictionaries to look up states for which the solution is known
    private static long inTheDict = 0;
    // the number of elements in the coincDictionary. If this value reaches
    // DICT_SIZE_MAX, a whole level is released from the dictionary and the
    // coincLevel is lowered by one
    private static HashMap<StateCode, Byte>[] minimaxDict = new HashMap[Main.MAX];
    // same for minimax. The difference is in that in the minimax dictionary the
    // actual minimax values are stored
    private static boolean terminateThreads = false; // True, only if a thread
    // has found a winning solution and all other threads should be terminated
    private static int reportsPrinted = 0; // Number of reports printed

    static {
        // Modifying the MINIMAX array
        for (byte i = 1; i < MINIMAX.length; i++)
            if ((MINIMAX[i] == 0) && (MINIMAX[i - 1] > 0))
                MINIMAX[i] = (byte) (MINIMAX[i - 1] - 1);
        for (byte i = 0; i < coincDict.length; i++) {
            coincDict[i] = new HashMap<>();
            minimaxDict[i] = new HashMap<>();
        }

        ITERATORS = new byte[TRANSFORMS.length][][];
        for (byte k = 0; k < TRANSFORMS.length; k++) {
            ITERATORS[k] = new byte[Main.MAX][];
            for (byte i = 0; i < Main.MAX_IND; i++)
                for (byte j = 0; j < Main.MAX_IND; j++)
                    ITERATORS[k][i * Main.MAX_IND + j] = TRANSFORMS[k].transform(i, j);
        }

    }

    private byte[][] board, fixed;
    private byte[] hor, ver, diaS, diaI, scores, fixedScores;
    private byte turn;
    private Byte minimaxScore;

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
     *
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
        if (terminateThreads)
            return Main.TRUCE;

        int level = scores[Main.WHITE] + scores[Main.DARK];
        if ((level == Main.MAX) || (reverse == 2) ||
                (fixedScores[Main.WHITE] > Main.MAX / 2) ||
                (fixedScores[Main.DARK] > Main.MAX / 2)) {
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

        StateCode curr = null;
        if (level <= coincLevel) {
            // check if the winner for this state was already calculated
            curr = getCode();
            if (level <= TRACE_LEVEL) { // for debug purposes only
                System.out.println(this);
            }
            if (coincDict[level - 1].get(curr) != null) {
                Main.coincCount[level - 1] += 1;
                return coincDict[level - 1].get(curr);
            }
        }

        count += 1;
        if ((level <= MULTIPROCESSING_LEVEL) &&
                ((count / REPORT_FQ) > reportsPrinted)) {
            reportsPrinted = (int) (count / REPORT_FQ);
            Main.printReport();
        }

        ArrayList<BoardState> moves;
        if (MINIMAX[level - 1] != 0) {
            moves = minimax((byte) (MINIMAX[level - 1] + level), null, (byte) 0, true, (byte) (level + MINIMAX_LEVELS_TO_STORE));
        } else
            moves = getMoves(true);

        /* for (BoardState move: moves)
            move.getCode(); */

        if ((moves == null) || (moves.size() == 0)) {
            turn = (byte) (1 - turn);
            return returnResult(curr, level, analyze(reverse + 1));
        }

        if (level == MULTIPROCESSING_LEVEL) {
            //TODO
        } else if (level < MULTIPROCESSING_LEVEL)
            Main.currBF[level - 1] = (long) moves.size();

        boolean trucePossible = false;
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
     * Does the necessary postprocessing of the result. Updates statistics, etc.
     *
     * @param curr
     * @param level
     * @param result
     * @return
     */
    private byte returnResult(StateCode curr, int level, byte result) {
        Main.levCount[level - 1] += 1;
        if (level <= coincLevel) {
            coincDict[level - 1].put(curr, result);
            inTheDict += 1;
            if (inTheDict > DICT_MAX_SIZE) {
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
                minimaxDict[i] = new HashMap<>();
                i += 1;
            }
        }
        if ((level == MULTIPROCESSING_LEVEL + 1) && (result == 1 - turn)) {
            //terminateThreads = true;
            //TODO:Turn this on
        }
        return result;
    }


    /**
     *
     Traverse the game tree down to the max_depth depth and then return the
     * list of possible moves in the order which guaranteers the best score
     * for the player on the max_depth depth.
     * WHITE tries minimizing, DARK tries maximizing
     * @param maxDepth
     * @param scoreAbove
     * @param reverse See the entry for analyze() for this one
     * @return
     */
    private ArrayList<BoardState> minimax(byte maxDepth, Byte scoreAbove,
                                          byte reverse, boolean sort, byte depthInDict) {
        minimaxScore = null;
        int level = scores[Main.WHITE] + scores[Main.DARK];
        if ((maxDepth == level) || (fixedScores[Main.WHITE] > Main.MAX / 2) ||
                (fixedScores[Main.DARK] > Main.MAX / 2) || (reverse == 2))
            return null;

        ArrayList<BoardState> moves = getMoves(false);

        if (moves.size() == 0) {
            turn = (byte) (1 - turn);
            minimax(maxDepth, null, (byte) (reverse + 1), false, depthInDict);
            turn = (byte) (1 - turn);
            return null;
        }

        for (BoardState move:moves) {
            byte currScore;
            StateCode tmp = move.getCode();
            //TODO check how many layers deep to go
            Byte dictEntry = null;
            if (level <= depthInDict)
                dictEntry = minimaxDict[level - 2].get(tmp);
            if (dictEntry != null) {
                currScore = minimaxDict[level - 2].get(tmp);
                move.minimaxScore = currScore;
            } else {
                move.minimax(maxDepth, minimaxScore, (byte) 0, false, depthInDict);
                currScore = move.getScore();
                if (level <= depthInDict)
                    minimaxDict[level - 2].put(tmp, currScore);
            }

            if ((minimaxScore == null) || ((turn == Main.WHITE) &&
                    (currScore < minimaxScore)) || ((turn == Main.DARK) &&
                    (currScore > minimaxScore)))
                minimaxScore = currScore;

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
        return result + (getCode()).toString() + "\n";
    }


    public String toString() {
        return toString((x, y) -> new byte[] {x, y});
    }

    /**
     * @param r
     * @param c
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

    /*private long getCode() {
        return getCode(true);
    }*/

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
    //private long getCode(boolean printIt) {
    private StateCode getCode() {
        //TODO is short enough for 8x8 board. Sign left shift
        ArrayList<Byte> its = new ArrayList<>();
        ArrayList<Byte> old_its;
        for (byte i = 0; i < ITERATORS.length; i++)
            its.add(i);
        StateCode code = new StateCode(turn);
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
                    its.add(j);
            }
            code.rows[i / Main.MAX_IND] = (short) ((code.rows[i / Main.MAX_IND] + max) * 3);
        }
        return code;
    }

    private byte getScore() {
        if (minimaxScore == null)
            return (byte) (fixedScores[Main.DARK] - fixedScores[Main.WHITE]);
        return minimaxScore;
    }

    /**
     * An interface for TRANSFORM lambda functions at the top
     */
    private interface Transformation {
        byte[] transform(byte x, byte y);
    }
}
