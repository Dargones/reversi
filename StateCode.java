package reversi;

import java.io.Serializable;

/**
 * Created by alexanderfedchin on 9/12/18.
 */
public class StateCode implements Serializable {
    // Represents the state as a code that can be hashed
    short[] rows;
    byte turn;

    StateCode(byte turn) {
        rows = new short[Main.MAX_IND];
        for (byte i = 0; i < rows.length; i++)
            rows[i] = 0;
        this.turn = turn;
    }

    public int hashCode() {
        int code = 0;
        for (byte i = 0; i < rows.length; i++)
            code += rows[i] % 32;
        return code + turn;
    }

    public String toString() {
        String result = "";
        for (byte i = 0; i < rows.length; i++)
            result += String.valueOf(rows[i]);
        return result + String.valueOf(turn);
    }

    public boolean equals(Object comp) {
        StateCode c = (StateCode) comp;
        for (byte i = 0; i < rows.length; i++)
            if (rows[i] != c.rows[i])
                return false;
        return turn == c.turn;
    }
}
