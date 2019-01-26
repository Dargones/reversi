package reversi;

import org.apache.log4j.Logger;

import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import static reversi.BoardState.*;
import static reversi.Disk.*;

/**
 * This classes processes the command arguments, launches the analysis and keeps the log, i.e.
 * nothing interesting is happenning here
 * Created by alexanderfedchin on 8/30/18.
 */
public class Main {
    private final static Logger logger = Logger.getLogger(StateAnalyzer.class);
    public static final byte INIT = 4;
    // total number of initial positions filled
    public static long count = 0; // number of different states considered while
    // traversing the game. The count does not include the final states
    public static Long[] levCount = new Long[MAX];
    // stores the total number of different states considered at each level
    // Level is the number of disks already on the board
    public static long[] lastTimeUpdated = new long[MAX];
    // the value of count at the last time a boardState at this level was calculated
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
     * Run the program and find the winner
     * @param args
     */
    public static void main(String args[]) {
        logger.info("Program launched");
        BoardState state = new BoardState();
        StateAnalyzer analyzer = new StateAnalyzer(state);
        Disk winner = analyzer.analyze();
        if (winner == DARK)
            System.out.println("Dark wins!");
        else if (winner == WHITE)
            System.out.println("White wins!");
        else
            System.out.println("Truce!");
        updateLog();
    }

    /**
     * Print report about the current progress. Current "time left" prediction works extremely
     * poorly
     */
    public static void updateLog() {
        logger.info("# of different states analyzed: " + sum(levCount));
        logger.info("# of states reused: " + sum(coincCount));
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
        logArray("Branching factors: ", bfs, INIT);
        logger.info("Min level reached: " + min_level_reached);

        double acceleration = 1;
        // how much faster the program is thanks to taking into account syjmetries and reflections
        for (byte i = INIT; i < MAX; i++)
            if (levCount[i] != 0)
                acceleration *= (double) (levCount[i] + coincCount[i]) / levCount[i];
        System.out.println("Acceleration due to reflection/rotation handling: " + acceleration);

        logArray("Current BFs: ", currBF, 1);
        if (min_level_reached == INIT) {
            // if this is the last time the function is called, save the coincDict
            try {
                for (int level = 3; level < StateAnalyzer.coincLevel; level++) {
                    FileOutputStream fileOut =
                            new FileOutputStream("BoardStates_level_" + level + ".ser");
                    ObjectOutputStream out = new ObjectOutputStream(fileOut);
                    out.writeObject(StateAnalyzer.coincDict[level]);
                    out.close();
                    fileOut.close();
                }
            } catch (Exception i) {
                logger.warn("Could not save the hashtables to the disk.");
            }
            return;
        }
        //TODO: Adequate remaining time prediction
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
    private static void logArray(String prompt, Number[] array, int beginID) {
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
        logger.info(indexLine);
        logger.info(arrayLine);
    }
}
