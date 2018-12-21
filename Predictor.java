package reversi;

import weka.classifiers.Classifier;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.SerializedInstancesLoader;
import weka.core.converters.SerializedInstancesSaver;
import weka.classifiers.Evaluation;

import java.io.File;
import java.util.Arrays;

/**
 * Created by alexanderfedchin on 12/15/18.
 * This class represents a classifier that can be used to predict the score of a given boardState
 */
public class Predictor {

    public static final double LEARNING_RATE = .03;
    public static final double MOMENTUM = .02;
    public static final String HIDDEN_LAYERS = "a";
    public static final int VALIDATION_SIZE = 11;
    public static final int VALIDATION_THRESHOLD = 3;
    public static final boolean DECAY = false;

    public static long timeStart = System.currentTimeMillis();
    // the last time classification process was launched
    private static Classifier[] classifiers = new Classifier[Main.MAX];
    // list of classifier for different levels of reversi boardstates.
    // Level is the number of disks already on the board
    // This array can only be updated by an external call from BoardState

    static {
        Arrays.fill(classifiers, null);
    }


    /**
     * This method is used by BoardState, whenever a new classifier is required for specific level.
     * A classifier is created by loading the trainig and testing sets from teh disk,
     * merging the two (since no testing is to be done) and then training a classifier. The data is
     * split in training and validation sets automatically
     * @param level
     * @return
     */
    public static Classifier getClassifier(int level) {
        if (classifiers[level] == null) {
            classifiers[level] =
                    createClassifier("data_level_" + level, "data_level_" + level + "_test");
            System.out.println("Classifier_" + level + " created");
        }
        return classifiers[level];
    }


    public static void main(String args[]) {
        int level = Integer.parseInt(args[0]);
        int evaluation_level = Integer.parseInt(args[1]);
        int size = Integer.parseInt(args[2]);
        createNewDataset(level, evaluation_level, size);
    }


    /**
     * Can be used to create new datasets. Datasets are created by randomly generating a
     * SIZE number of BoardStates on specific LEVEL. Minimax is then used to evaluate the
     * BoardStates. Evaluation Level can be Main.MAX + 1, in which case the actual value of
     * the state is assessed. Otherwise, a pretrained model for the evaluation state is used
     * @param level            level for which the dataset is created
     * @param evaluationLevel  level of evaluation of minimax. If this parameter is equal to
     *                         Main.MAX + 1, the actual values of states will be assessed
     * @param size             Number of BoardStates in a dataset
     */
    public static void createNewDataset(int level, int evaluationLevel, int size) {
        System.out.println("Creating a dataset for level " + level);
        BoardState state = new BoardState();
        Classifier classifier = null;
        Instances training_set, testing_set;
        System.out.println("Program launched");
        if (evaluationLevel != Main.MAX + 1) {
            classifier = createClassifier("data_level_" + evaluationLevel,
                    "data_level_" + evaluationLevel + "_test");
            System.out.println("Model trained");
        }
        timeStart = System.currentTimeMillis();
        training_set = state.getInstances((byte) level, size / 10 * 9,
                "data_level_" + level, evaluationLevel, classifier);
        timeStart = System.currentTimeMillis();
        testing_set = state.getInstances((byte) level, size / 10,
                "data_level_" + level +"_test", evaluationLevel, classifier);
        saveInstances(training_set, "data_level_" + level);
        saveInstances(testing_set, "data_level_" + level + "_test");
        System.out.println("Training a a classifier for level " + level);
        Classifier classifier1 = train(LEARNING_RATE, HIDDEN_LAYERS, VALIDATION_SIZE,
                VALIDATION_THRESHOLD, DECAY, MOMENTUM, training_set);
        System.out.println("Testing the classifier");
        test(classifier1, testing_set, false);
        System.out.println("Testing the 'unflippable' method");
        test(classifier1, testing_set, true);
    }

    /**
     * Train a classifier on a dataset that is already created. This classifier is not for testing,
     * but for evaluation
     * @param file1  training set (is merged with the testing set, because no testing is done)
     * @param file2  testing set (is merged with the training set, because no testing is done)
     * @return
     */
    public static Classifier createClassifier(String file1, String file2) {
        Instances set = loadInstances(file1);
        Instances additional = loadInstances(file2);
        for (int i = 0; i < additional.numInstances(); i++)
            set.add(additional.instance(i));
        return train(LEARNING_RATE, HIDDEN_LAYERS, VALIDATION_SIZE,
                VALIDATION_THRESHOLD, DECAY, MOMENTUM, set);
    }


    /**
     * Train a model using a specified training set and a selected set of parameters
     * @param learningRate
     * @param hiddenLayers
     * @param validationSize
     * @param validationThreshold
     * @param decay
     * @param momentum
     * @param training_set
     * @return
     */
    public static Classifier train(double learningRate, String hiddenLayers,
                                   int validationSize, int validationThreshold,
                                   boolean decay, double momentum,
                                   Instances training_set) {
        MultilayerPerceptron classifier = new MultilayerPerceptron();
        classifier.setValidationSetSize(validationSize);
        classifier.setDecay(decay);
        classifier.setValidationThreshold(validationThreshold);
        classifier.setLearningRate(learningRate);
        classifier.setHiddenLayers(hiddenLayers);
        classifier.setMomentum(momentum);
        System.out.println("Model built");
        try {
            classifier.buildClassifier(training_set);
        } catch (Exception ex) {
            throw new RuntimeException();
        }
        return classifier;
    }

    /**
     * Test a classifier using F-score. Using Area under the ROC curve might be a better way to do
     * evalute the model, but it is unclear how to calculate probabilities from the score, and
     * nothing can be done about draws (since they are, effectively, the third class)
     * @param classifier
     * @param testing_set
     * @param testUnflippable if true, test the unflippable method instead of the classifier itself
     * @return
     */
    public static double test(Classifier classifier, Instances testing_set, boolean testUnflippable) {

        int[][] stats = new int[3][];
        for (int i = 0; i < stats.length; i++) {
            stats[i] = new int[3]; // confusion matrix
            Arrays.fill(stats[i], 0);
        }

        try {
            for (int i = 0; i < testing_set.numInstances(); i++) {
                Instance wekaInstance = testing_set.instance(i);
                int targetIndex;
                if (testUnflippable)
                    targetIndex = (int) (wekaInstance.value(2)-wekaInstance.value(3));
                    // TODO change teh to numbers to constants
                else
                    targetIndex = (int) classifier.classifyInstance(wekaInstance);
                int actualIndex = (int) wekaInstance.value(BoardState.attributes.size() - 1);
                // building the confusion matrix
                if (targetIndex > 0) {
                    if (actualIndex > 0)
                        stats[1][1] += 1;
                    else if (actualIndex < 0)
                        stats[0][1] += 1;
                    else
                        stats[2][1] += 1;
                } else if (targetIndex < 0) {
                    if (actualIndex > 0)
                        stats[1][0] += 1;
                    else if (actualIndex < 0)
                        stats[0][0] += 1;
                    else
                        stats[2][0] += 1;
                } else {
                    if (actualIndex > 0)
                        stats[1][2] += 1;
                    else if (actualIndex < 0)
                        stats[0][2] += 1;
                    else
                        stats[2][2] += 1;
                }
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        for (int i = 0; i < stats.length; i++)
            System.out.println(Arrays.toString(stats[i])); // print the confusion matrix
        double fScore0 = fScore(stats[0][0], stats[1][0] + stats[2][0], stats[0][1] + stats[0][2]);
        double fScore1 = fScore(stats[1][1], stats[0][1] + stats[2][1], stats[1][0] + stats[1][2]);
        double fScore2 = fScore(stats[2][2], stats[1][2] + stats[0][2], stats[2][1] + stats[2][0]);
        double meanfScore = (fScore0 * (stats[0][0] + stats[0][1] + stats[0][2]) +
                fScore1 * (stats[1][0] + stats[1][1] + stats[1][2]) +
                fScore2 * (stats[2][0] + stats[2][1] + stats[2][2])) / testing_set.numInstances();
        System.out.println("F scores: "+ fScore0 + ", " + fScore1 + ", " + fScore2 +
                ", mean:" + meanfScore + "\n");
        return meanfScore;
    }

    /**
     * Calculate teh f-score given TP, FP, and FN
     * @param tp
     * @param fp
     * @param fn
     * @return
     */
    public static double fScore(int tp, int fp, int fn) {
        if (tp == 0)
            return 0;
        double precision = (double) (tp) / (tp + fp);
        double recall = (double) (tp) / (tp + fn);
        return 2 * (recall * precision) / (recall + precision);
    }

    /**
     * Saves a dataset to a file
     * @param instances
     * @param filename
     */
    public static void saveInstances(Instances instances, String filename) {
        SerializedInstancesSaver saver = new SerializedInstancesSaver();
        try {
            saver.setFile(new File(filename));
            saver.setInstances(instances);
            saver.writeBatch();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Loads a dataset from a file
     * @param filename
     */
    public static Instances loadInstances(String filename) {
        SerializedInstancesLoader loader = new SerializedInstancesLoader();
        try {
            loader.setFile(new File(filename));
            return loader.getDataSet();
        } catch (Exception e) {
            return null;
        }
    }
}
