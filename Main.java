package reversi;

import weka.classifiers.Classifier;

import java.io.FileOutputStream;
import java.io.ObjectOutputStream;

/**
 * Created by alexanderfedchin on 8/30/18.
 */
public class Main {
    public static final byte WHITE = 0;
    public static final byte DARK = 1;
    public static final byte TRUCE = 2; // can also mean "unoccupied"
    public static final byte MAX_IND = 6; // the dimension of the board.
    // The ultimate goal is to solve the puzzle for MAX_IND = 8
    public static final byte MAX = MAX_IND * MAX_IND;
    // total number of tiles on the board
    public static final byte INIT = 4;
    // total number of initial positions filled
    public static long count = 0; // number of different states considered while
    // traversing the game. The count does not include the final states
    public static Long[] levCount = new Long[MAX];
    // stores the total number of different states considered at each level
    // Level is the number of disks already on the board
    public static long[] lastTimeUpdated = new long[Main.MAX];
    // the value of count at the last time a boardstate at this level was calculated
    public static Long[] coincCount = new Long[MAX];
    // stores the number of times the coincDict was used. Either levCount or
    // coincCount can be updated at each pass, but never both. This array is used to assess the
    // benefit of using a dictionary to store previously evaluated states
    public static Long[] currBF = new Long[MAX];
    // current branching factors
    public static long timeStart = System.currentTimeMillis();

    static {
        for (byte i = 0; i < MAX; i ++) {
            levCount[i] = 0L;
            coincCount[i] = 0L;
            currBF[i] = 0L;
            lastTimeUpdated[i] = 0L;
        }
    }

    /**
     * Run t heprogram and find the winner
     * @param args
     */
    public static void main(String args[]) {
        BoardState state = new BoardState();
        byte winner = state.analyze();
        if (winner == DARK)
            System.out.println("Dark wins!");
        else if (winner == WHITE)
            System.out.println("White wins!");
        else
            System.out.println("Truce!");
        printReport();
    }

    /**
     * Print report about the current progress. Current "time left" prediction works extremely
     * poorly
     */
    public static void printReport() {
        System.out.println("# of different states analyzed: " + sum(levCount));
        System.out.println("# of states reused: " + sum(coincCount));
        Double[] bfs = new Double[MAX - INIT]; // branching factors
        byte min_level_reached = 0;
        for (byte i = INIT; i < MAX; i++) {
            if (levCount[i - 1] == 0)
                bfs[i - INIT] = 0.;
            else {
                if (min_level_reached == 0)
                    min_level_reached = i;
                bfs[i - INIT] = ((double) levCount[i] / levCount[i - 1]);
            }
        }
        printArray("Branching factors: ", bfs, INIT);
        System.out.println("Min level reached: " + min_level_reached);

        double acceleration = 1;
        // how much faster the program is thanks to taking into account syjmetries and reflections
        for (byte i = INIT; i < MAX; i++)
            if (levCount[i] != 0)
                acceleration *= (double) (levCount[i] + coincCount[i]) / levCount[i];
        System.out.println("Acceleration: " + acceleration);

        printArray("Current BFs: ", currBF, 1);
        if (min_level_reached == INIT) {
            // if this is the last time the function is called, save teh coincDict
            try {
                for (int level = 3; level < BoardState.coincLevel; level++) {
                    FileOutputStream fileOut =
                            new FileOutputStream("hashtable" + level + ".ser");
                    ObjectOutputStream out = new ObjectOutputStream(fileOut);
                    out.writeObject(BoardState.coincDict[level]);
                    out.close();
                    fileOut.close();
                }
            } catch (Exception i) {
                i.printStackTrace();
            }
            return;
        }

        Long statesLeft = 0L;
        // states left to analyze before next level is reached
        boolean non_zero_found = false;
        // whether the first non-zero element of currBF was reached
        for (int i = MAX - 2; i >= min_level_reached - 3; i--) {
            double curr = currBF[i];
            if (non_zero_found)
                curr -= 1;
            if (curr != 0)
                non_zero_found = true;
            curr *= (double) lastTimeUpdated[i + 1] / levCount[i + 1] * ((double) sum(levCount) / count);
            statesLeft += (long) curr;
        }
        double totalMod = (double) (statesLeft + sum(levCount)) / statesLeft;
        // totalMod ties the states left (before next level) and the total
        // expected number of states
        for (int i = min_level_reached - 3; i > 2; i--) {
            totalMod *= currBF[i];
        }
        long currTime = System.currentTimeMillis();
        long timeLeft = (long) ((double) (currTime - timeStart) / sum(levCount) * statesLeft);

        System.out.println(statesLeft + " states left before next level");
        System.out.println(((Double)(totalMod * statesLeft)).longValue() +
                " states left total");
        System.out.println("Time left before next level: " + getDuration(timeLeft));
        System.out.println("Time left total: " + getDuration((long) (timeLeft * totalMod)));
        System.out.println("Current time: " + currTime);
        System.out.println("\n");
    }

    /**
     * Convert miliseconds to a human-readable String representing time
     * @param milisec
     * @return
     */
    public static String getDuration(long milisec) {
        long total = milisec / 1000;
        long sec = total % 60;
        total = total / 60;
        long min = total % 60;
        total = total / 60;
        long hour = total % 24;
        total = total / 24;
        long day = total % 365;
        total = total / 365;
        return total + " years, " + day + " days, " + hour + ':' + min + ":" + sec;
    }

    /**
     * Returns the sum of all elements in the array
     * @param array
     * @return
     */
    public static long sum(Number[] array) {
        long result = (long) array[0];
        for (int i = 1; i < array.length; i ++)
            result += (long) array[i];
        return result;
    }

    /**
     * Returns the sum of all elements in the array
     * @param array
     * @return
     */
    public static long sum(byte[] array) {
        long result = array[0];
        for (int i = 1; i < array.length; i++)
            result += array[i];
        return result;
    }

    /**
     * Get a nice String representation of an array
     * @param prompt
     * @param array
     * @param beginID
     * @return
     */
    private static void printArray(String prompt, Number[] array, int beginID) {
        String indexLine = prompt;
        String arrayLine = "";
        for (int i = 0; i < prompt.length(); i++)
            arrayLine += " ";
        for (int i = 0; i < array.length; i++) {
            arrayLine += array[i].toString() + " ";
            String tmp = ((Integer) (i + beginID)).toString();
            int len = array[i].toString().length() + 1;
            for (int j = tmp.length(); j < len; j++)
                tmp += " ";
            indexLine += tmp;
        }
        System.out.println(indexLine);
        System.out.println(arrayLine);
    }
}
