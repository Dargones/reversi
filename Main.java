package reversi;

import weka.classifiers.Classifier;

/**
 * Created by alexanderfedchin on 8/30/18.
 */
public class Main {
    public static final byte WHITE = 0;
    public static final byte DARK = 1;
    public static final byte TRUCE = 2; // can also mean "unoccupied"
    public static final byte MAX_IND = 6; // the dimension of the board.
    // The goal is to solve the puzzle for MAX_IND = 8
    public static final byte MAX = MAX_IND * MAX_IND;
    // total number of tiles on the board
    public static final byte INIT = 4;
    // initial positions filled
    public static final double[] EXP_BF = {0, 0, 0, 1.0, 3.0, 1.33, 3.25, 1.15,
            4.67, 1.03, 4.54, 1.18, 4.12, 1.25, 4.21, 1.45, 0.0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    // experimentally obtained branching factors for lower levels - these can be
    // used to predict the time remaining to solve the problem. 0 means that
    // there is no experimentally obtained value

    public static Long[] levCount = new Long[MAX];
    // stores the total number of different states considered at each level
    public static Long[] coincCount = new Long[MAX];
    // stores the number of times the coincDict was used. Either levCount or
    // coincCount can be updated at each pass, but never both
    public static Long[] currBF = new Long[MAX];
    // current branching factors
    public static long timeStart = System.currentTimeMillis();

    static {
        for (byte i = 0; i < MAX; i ++) {
            levCount[i] = 0L;
            coincCount[i] = 0L;
            currBF[i] = 0L;
        }
    }

    public static void main(String args[]) {
        Classifier classifier25 = Predictor.getClassifier("data_level_25", "data_level_25_test");
        System.out.println("Classifier-25 loaded");
        /*Classifier classifier20 = Predictor.getClassifier("data_level_20", "data_level_20_test");
        System.out.println("Classifier-20 loaded");
        Classifier classifier18 = Predictor.getClassifier("data_level_18", "data_level_18_test");
        System.out.println("Classifier-18 loaded");*/
        for (int i = 24; i > 0; i--)
            BoardState.MINIMAX_CLASS[i] = classifier25;
        /*for (int i = 19; i > 0; i--)
            BoardState.MINIMAX_CLASS[i] = classifier20;
        for (int i = 17; i > 0; i--)
            BoardState.MINIMAX_CLASS[i] = classifier18;*/
        BoardState state = new BoardState();
        //TODO: find an alternative for cProfile
        byte winner = state.analyze();
        if (winner == DARK)
            System.out.println("Dark wins");
        else if (winner == WHITE)
            System.out.println("White wins");
        else
            System.out.println("Truce");
        printReport();
    }

    /**
     * Print report about the current progress
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
        // how much faster the program is thanks to transforms
        for (byte i = INIT; i < MAX; i++)
            if (levCount[i] != 0)
                acceleration *= (double) (levCount[i] + coincCount[i]) / levCount[i];
        System.out.println("Acceleration: " + acceleration);

        printArray("Current BFs: ", currBF, 1);
        if (min_level_reached == INIT)
            return;

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
            for (int j = i + 2; j < MAX; j++)
                curr *= bfs[j - INIT];
            statesLeft += (long) curr;
        }
        double totalMod = (statesLeft + sum(levCount)) / statesLeft;
        // totalMod ties the states left (before next level) and the total
        // expected number of states
        for (int i = min_level_reached - 3; i > 2; i--)
            if (EXP_BF[i] != 0)
                totalMod *= EXP_BF[i];
            else
                totalMod *= currBF[i];
        long timeLeft = (long) ((double) (System.currentTimeMillis() - timeStart) /
                sum(levCount) * statesLeft);

        System.out.println(statesLeft + " states left before next level");
        System.out.println(((Double)(totalMod * statesLeft)).longValue() +
                " states left total");
        System.out.println("Time left before next level: " + getDuration(timeLeft));
        System.out.println("Time left total: " + getDuration((long) (timeLeft * totalMod)));
        System.out.println("\n");
        /* try {
            Thread.sleep(20000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }*/
    }

    private static String getDuration(long milisec) {
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
