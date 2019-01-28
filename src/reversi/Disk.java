package reversi;

import java.io.Serializable;

/***
 * Simple enum for three possible states in which a cell on a board can be. The cell can either
 * have a WHITE disk on it, a DARK one, or NONE at all. The same enum is used for annotating the
 * winner (either WHITE wins or DARK wins, or NONE of the two is the winner)
 */
public enum Disk implements Serializable {
    WHITE((byte) 0,'O'), DARK((byte) 1, 'X'), NONE((byte) 2, '.');

    public final byte id;    // unique id associated with the enum value (used for indexing)
    public final char name; // a char representation used in toString()

    Disk(byte id, char name) {
        this.id = id;
        this.name = name;
    }

    public static Disk getReverse(Disk current) {
        if (current == WHITE)
            return DARK;
        if (current == DARK)
            return WHITE;
        return NONE;
    }
}
