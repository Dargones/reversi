package reversi;

import java.io.Serializable;
import java.util.*;
import static reversi.Disk.*;

/**
 * Created by alexanderfedchin on 8/31/18.
 * This class represents a BoardState. It stores the positions of disks on the board. It also stores
 * the information about which player is to make the next move. Some basic functionality
 * (like getting a list of available moves) are present as well
 */
public class BoardState implements Serializable {

    public static final byte DIM = 6; // the dimension of the board.
    // The ultimate goal is to solve the game for DIM = 8
    public static final byte MAX = DIM * DIM;
    // total number of tiles on the board
    private static final Coord[] DIRS = {new Coord(0, 1), new Coord(1, 1),
            new Coord(1, 0), new Coord(1, -1),
            new Coord(0, -1), new Coord(-1, -1),
            new Coord(-1, 0), new Coord(-1, 1)};
    // the 8 directions in which one can go from a tile.
    private static final Transformation[] TRANSFORMS = {
            Coord::new, (x, y) -> new Coord(DIM - 1 - x, y),
            (x, y) -> new Coord(x, DIM - 1 - y), (x, y) -> new Coord(y, x),
            (x, y) -> new Coord(DIM - 1 - y, DIM - 1 - x),
            (x, y) -> new Coord(DIM - 1 - x, DIM - 1 - y),
            (x, y) -> new Coord(y, DIM - 1 - x),
            (x, y) -> new Coord(DIM - 1 - y, x)};
    // These lambda functions represent the four rotations and the four
    // reflections which can be applied to a BoardState. The functions are only
    // used inside the static block and during the debug process.
    private static final Coord[][] ITERATORS;
    // 8 arrays of coordinates, which correspond to 8 transformations. ITERATIONS[0] is an array of
    // all possible coordinates arranged arbitrary. For all i > 0, ITERATORS[i][j] stores the
    // coordinates which one gets by applying TRANSFORMS[i] to ITERATORS[0][j]
    private static final byte AVG_MOVES_POSSIBLE = 3; // average branching factor. Used for initial
    // capacity in the getMoves() method. Determined empirically

    static {
        // Building the ITERATORS Arrays using TRANSFORMS
        ITERATORS = new Coord[TRANSFORMS.length][];
        for (byte k = 0; k < TRANSFORMS.length; k++) {
            ITERATORS[k] = new Coord[MAX];
            for (byte i = 0; i < DIM; i++)
                for (byte j = 0; j < DIM; j++)
                    ITERATORS[k][i * DIM + j] = TRANSFORMS[k].transform(i, j);
        }
    }

    public byte[] scores; // scores[WHITE.id] - number of white scores on board,
    // scores[DARK.id] - number of dark disks on the board
    private byte[][] board;
    // board is a two dimensional array where each element is an id of a Disk instance. The Disk
    // instances themselves are not stored, because this would be memory inefficient.
    private Disk turn; // the player to make the next move. Either Disk.WHITE.id or Disk.DARK.id
    private long[] code = null; // each two different boardStates have identical codes, unless they
    // are rotations or reflections of each other (in which case they are considered equal). The
    // code is stored in a long array (see getCode()). BigInteger is too slow.
    // TODO: it might eventually make sense to make a separate class for that, but having a separate
    // class bight still be too slow. The code is initialized to null

    /**
     * Default constructor
     */
    public BoardState() {
        // creating an empty board
        board = new byte[DIM][DIM];
        for (byte i = 0; i < board.length; i++)
            Arrays.fill(board[i], NONE.id);

        // the initial state of the board according to the Reversi rules
        scores = new byte [] {(byte) 2, (byte) 2};
        int b = DIM / 2;
        int a = b - 1;
        board[a][a] = WHITE.id;
        board[a][b] = DARK.id;
        board[b][b] = WHITE.id;
        board[b][a] = DARK.id;
        turn = DARK;
    }

    /**
     * Create a copy of an original BoardState.
     * @param original
     */
    public BoardState (BoardState original) {
        board = new byte[DIM][];
        for (byte i = 0; i < board.length; i++)
            board[i] = Arrays.copyOf(original.board[i], DIM);
        turn = original.turn;
        scores =  Arrays.copyOf(original.scores, 2);
    }

    /**
     * Get the list of possible moves (boards)
     *
     * @param sort if True, sort the moves by the number of disks the current player will have
     *             after executing the move
     */
    public ArrayList<BoardState> getMoves(boolean sort) {
        ArrayList<BoardState> moves = new ArrayList<>(AVG_MOVES_POSSIBLE);
        for (byte i = 0; i < board.length; i++)
            for (byte j = 0; j < board.length; j++) {
                if (board[i][j] != NONE.id)
                    continue;
                // the tile is not occupied but it is yet to be discovered,
                // whether something can be placed here
                BoardState trial = tryMove(i, j);
                if (trial != null)
                    moves.add(trial);
            }
        if (sort)
            moves.sort((x, y) -> y.getScoreDifference() - x.getScoreDifference());
        return moves;
    }

    /**
     * Attempt to place a disk at (row, column). If this is possible, create a
     * new object describing the changes made to the board. To understand this function it is
     * necessary to understand Reversi game rules (https://en.wikipedia.org/wiki/Reversi - Modern
     * Version)
     *
     * @param r the row where the disk is to be placed
     * @param c the column where the disk is to be placed
     * @return
     */
    private BoardState tryMove(byte r, byte c) {
        BoardState result = null;
        for (Coord dir : DIRS) { // for every possible direction
            boolean disksToFlip = false; //there are disks of the opposite color to be flipped
            boolean disksEnclosed = false; // the disks that are to be flipped are enclosed
            // between the newly placed disk and some other disk of the same color
            byte m = 1; // multiplier
            while ((r + dir.r * m < DIM) && (r + dir.r * m >= 0) &&
                    (c + dir.c * m < DIM) && (c + dir.c * m >= 0)) {
                if (board[r + dir.r * m][c + dir.c * m] == turn.id) {
                    disksEnclosed = true;
                    break;
                } else if (board[r + dir.r * m][c + dir.c * m] == NONE.id)
                    break;
                else
                    disksToFlip = true;
                m += 1;
            }
            if ((!disksEnclosed) || (!disksToFlip))
                continue;

            if (result == null) { // if the new BoardState was not created yet
                result = new BoardState(this);
                result.board[r][c] = turn.id;
                result.scores[turn.id] += 1;
                result.reverseTurn();
            }

            for (m = 1; board[r + dir.r * m][c + dir.c * m] != turn.id; m++) {// flipping disks
                result.board[r + dir.r * m][c + dir.c * m] = turn.id;
                result.scores[turn.id] += 1;
                result.scores[1 - turn.id] -= 1;
            }
        }
        return result;
    }

    /**
     * Get a string representation of the state
     * X corresponds to dark disks
     * 0 corresponds to white disks
     * . corresponds to unoccupied tiles
     *
     * @param transform Transformation to apply to teh board before getting its representation
     * @return
     */
    private String toString(Transformation transform) {
        String result = "";
        Coord coord;
        for (byte i = 0; i < board.length; i++) {
            for (byte j = 0; j < board[i].length; j++) {
                coord = transform.transform(i, j);
                if (board[coord.r][coord.c] == DARK.id)
                    result += DARK.name;
                else if (board[coord.r][coord.c] == WHITE.id)
                    result += WHITE.name;
                else
                    result += NONE.name;
            }
            result += "\n";
        }
        return result + "Code: " + Arrays.toString((getCode())) + "\n";
    }

    /**
     * The default version of the toString() method that does not rotate or reflect the board
     * before printing it
     * @return
     */
    public String toString() {
        return toString(Coord::new);
    }

    /**
     * This implementation of equals makes the two board equal if they are one rotation or
     * reflection from each other
     * @param other
     * @return
     */
    public boolean equals(Object other) {
        if (other == null)
            return false;
        for (int i = 0; i < getCode().length; i++)
            if (getCode()[i] != ((BoardState) other).getCode()[i])
                return false;
        return true;
    }


    /**
     * Hash code that uses the code as a unique identifier
     * @return
     */
    public int hashCode() {
        return Arrays.hashCode(getCode());
    }

    /**
     * A getter
     * @return
     */
    public byte[][] getBoard() {
        return board;
    }

    /**
     * A getter
     */
    public Disk getTurn() {
        return turn;
    }

    /**
     * Level is the total number of disks on the board. Once level == MAX, the game is finished
     * (although it can finish before that as well if no player can make a move)
     * @return
     */
    public int getLevel() {
        return scores[WHITE.id] + scores[DARK.id];
    }

    /**
     * Reverses the turn variable
     */
    public void reverseTurn() {
        turn = getReverse(turn);
    }

    /**
     * Returns the difference between scores[turn.id] and scores[getReverse(turn).id]
     * @return
     */
    public int getScoreDifference() {
        return scores[turn.id] - scores[getReverse(turn).id];
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
     * @return the code
     */
    public long[] getCode() {
        if (code != null)
            return code;
        ArrayList<Byte> its = new ArrayList<>(); // iterators' ids
        ArrayList<Byte> old_its; // the copy of old iterators' ids
        for (byte i = 0; i < ITERATORS.length; i++)
            its.add(i);
        // filling the ids arrays. Initially, no iterator sequence maximizes the code
        code = new long[2]; // this function must work with DIM = 8.
        // Thus, the code can have 3^(8^2) * 2 (2 is for the turn variable) values. On practice, I
        // use two longs (128 bits)
        for (int i = 0; i < MAX; i++) {
            byte max = -1;
            old_its = its;
            for (byte j = 0; j < old_its.size(); j++) {
                Coord coords = ITERATORS[old_its.get(j)][i];
                if (board[coords.r][coords.c] > max) {
                    max = board[coords.r][coords.c];
                    its = new ArrayList<>(1);
                }
                if (board[coords.r][coords.c] == max)
                    its.add(old_its.get(j));
            }
            code[i % 2] = (code[i % 2] + max) * 3;
        }
        code[code.length - 1] += turn.id;
        return code;
    }

    /**
     * A class representing a coordinate (pair of ints)
     */
    private static class Coord {
        public int r, c; // row and column

        Coord(int r, int c) {
            this.r = r;
            this.c = c;
        }
    }

    /**
     * An interface for TRANSFORM lambda functions at the top (See comments there)
     */
    private interface Transformation {
        Coord transform(int x, int y);
    }
}
