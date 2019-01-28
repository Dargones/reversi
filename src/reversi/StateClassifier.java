package reversi;

import static reversi.BoardState.*;
import static reversi.Disk.*;

import org.apache.log4j.Logger;
import weka.classifiers.Classifier;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.*;
import weka.core.converters.SerializedInstancesLoader;
import weka.core.converters.SerializedInstancesSaver;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

/**
 * Created by alexanderfedchin on 12/15/18.
 * This class represents a classifier that can be used to predict the score of a given boardState
 */
public class StateClassifier {

    private static final Logger logger = Logger.getLogger(reversi.StateClassifier.class);
    private static final String DATASETS_DIR = "dataSets/";
    // directory in which to store the datasets
    private static final String MODELS_DIR = "models/";
    // directory in which to store the models
    private static final byte[] MINIMAX_EVALUATORS = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 25, 0, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0};
    // For each level i, this array stores the level at which the minimax launched from level i
    // should be evaluated. This is used in getDataSet()
    private static final int DATASET_SIZE = 20000; // default dataSet size
    private static final int TESTING_SET_SIZE = (int) (DATASET_SIZE * 0.1);
    private static StateClassifier[] classifiers = new StateClassifier[MAX];
    // list of classifier for different levels.
    // Level is the number of disks already on the board
    public static ArrayList<Attribute> attributes = new ArrayList(); // required for Weka to work
    private static Instances evaluationSet; // required for Weka to work

    static {
        Arrays.fill(classifiers, null);
        for (byte i = 0; i < MAX; i++)
            attributes.add(new Attribute(String.valueOf(i)));
        Attribute label = new Attribute("label", MAX);
        attributes.add(label);
    }

    private Classifier classifier; // the under-the-hood classifier

    /**
     * Default constructor
     * @param classifier
     */
    public StateClassifier(Classifier classifier) {
        this.classifier = classifier;
        // Creating an evaluation set is required even when evaluating instances one by one
        evaluationSet = new Instances("evaluationSet", attributes, 1);
    }

    /**
     * A constructor that creates a classifier from a given trainig set
     * @param dataSet
     */
    public StateClassifier(Instances dataSet) {
        this(trainWekaClassifier(dataSet));
    }

    /**
     * For testing and debugging only
     * @param args
     */
    public static void main(String[] args) {
        // this sequence of levels (and also MINIMAX_EVALUATORS values) is (empirically) the best
        //testClassifier(25);
        //testClassifier(20);
        testClassifier(18);
    }

    /**
     * Train a model using a specified training set. The current model is a demo version
     * (feed-forward NN with one hidden layer) for what
     * will in future be replaced by a CNN model based on https://arxiv.org/pdf/1711.06583.pdf
     * @return
     */
    private static Classifier trainWekaClassifier(Instances dataSet) {
        MultilayerPerceptron classifier = new MultilayerPerceptron();
        classifier.setValidationSetSize(10);
        classifier.setDecay(false);
        // with decay = true, the training is painfully slow and for a demo version false is OK
        classifier.setValidationThreshold(2);
        classifier.setLearningRate(.02);
        classifier.setHiddenLayers("a,a"); // a = # of attributes + # of classes
        classifier.setMomentum(.02);
        logger.info("Model built");
        try {
            classifier.buildClassifier(dataSet);
        } catch (Exception ex) {
            logger.error("Model training failed");
            System.exit(-1);
        }
        return classifier;
    }

    /**
     * Classify a boardState and return a result (the fuction is called StateAnalyzer.minimax)
     * @param state the state to classify
     * @return
     */
    public byte classify(BoardState state) {
        //The score returned should be between -MAX, MAX
        //The score returned should be positive if the current player is winning
        evaluationSet.clear();
        Instance instance = createInstance(state, null);
        evaluationSet.add(instance);
        try {
            return (byte) Math.round(classifier.classifyInstance(instance));
        } catch (Exception e) {
            logger.error("Instance classification failed. Consider terminating the program");
            return 0;
        }
    }

    /**
     * Attempt to load a requested model from the disk. If the model is already loaded to memory,
     * simply return it. If the model is neither in the disk nor in memory, train a
     * new one. Return the corresponding classifier.
     * @param level Level of the model (level is the number of disks on the board)
     * @return
     */
    public static StateClassifier getStateClassifier(int level) {
        if (classifiers[level - 1] != null)
            return classifiers[level - 1];
        StateClassifier classifier;
        logger.info("Attempting to load a pretrained model for level " + level);
        try {
            classifier = new StateClassifier((Classifier) SerializationHelper.read(
                    MODELS_DIR + "Model_level_" + level + ".ser"));
            logger.info("Model loaded");
            classifiers[level - 1] = classifier;
            return classifiers[level - 1];
        } catch (Exception e) {
            logger.info(e.fillInStackTrace() + " Loading failed. Attempting to load a " +
                    "corresponding dataSet to train a new model");
            Instances dataSet = getDataset(level);
            classifier = new StateClassifier(dataSet);
            logger.info("Model trained. Saving the model");
        }
        try {
            SerializationHelper.write(
                    MODELS_DIR + "Model_level_" + level + ".ser", classifier.classifier);
        } catch (Exception e2) {
            logger.warn("Model could not be saved");
        }
        classifiers[level - 1] = classifier;
        return classifiers[level - 1];
    }

    /**
     * Attempt to load a requested dataSet from the disk. If the dataset cannot be loaded, create a
     * new one.
     * @param level
     * @return
     */
    public static Instances getDataset(int level) {
        SerializedInstancesLoader loader = new SerializedInstancesLoader();
        try {
            loader.setFile(new File(DATASETS_DIR + "Data_level_" + level + ".ser"));
            return loader.getDataSet();
        } catch (IOException e) {
            logger.info("Loading failed. Creating a new dataset with default configurations");
            return createDataset(level);
        }
    }

    /**
     * Create a new dataset (for a given level) using StateAnalyzer.createDataset
     * @param level
     * @return
     */
    public static Instances createDataset(int level) {
        StateAnalyzer.DataSet rawData;
        int evaluationLevel = MINIMAX_EVALUATORS[level - 1];
        if (evaluationLevel == 0)
            rawData = StateAnalyzer.createDataset(level, DATASET_SIZE, MAX, null);
        else
            rawData = StateAnalyzer.createDataset(level, DATASET_SIZE, evaluationLevel,
                    getStateClassifier(evaluationLevel));

        Instances dataSet = new Instances("Data_level" + level, attributes, DATASET_SIZE);
        for (int i = 0; i < rawData.instances.length; i++)
            dataSet.add(createInstance(rawData.instances[i], rawData.labels[i]));
        dataSet.setClassIndex(MAX);
        saveDataSet(dataSet, DATASETS_DIR + "Data_level_" + level + ".ser");
        return dataSet;
    }

    /**
     * Save a dataset to a file
     * @param instances
     * @param filename
     */
    public static void saveDataSet(Instances instances, String filename) {
        SerializedInstancesSaver saver = new SerializedInstancesSaver();
        try {
            saver.setFile(new File(filename));
            saver.setInstances(instances);
            saver.writeBatch();
        } catch (Exception e) {
            logger.warn("Saving dataSet to disk failed");
        }
    }

    /**
     * Convert a BoardState and its evaluation to a weka Instance
     * @param state
     * @param label
     * @return
     */
    public static Instance createInstance(BoardState state, Byte label) {
        Instance instance = new DenseInstance(attributes.size());
        for (byte i = 0; i < DIM; i++)
            for (byte j = 0; j < DIM; j++) {
                int value;
                if (state.getBoard()[i][j] == state.getTurn().id)
                    value = 1;
                else
                    value = ((state.getBoard()[i][j] == NONE.id) ? 0 : -1);
                instance.setValue(i * DIM + j, value);
            }
        if (label != null)
            instance.setValue(MAX, label);
        return instance;
    }

    /**
     * Test a classifier using F-score. Using Area under the ROC curve might be a better way to do
     * evaluate the model, but it is unclear how to calculate probabilities from the score, and
     * nothing can be done about draws (since they are, effectively, the third class)
     * @return
     */
    public static double testClassifier(int level) {

        Instances dataSet = getDataset(level);
        Instances testingSet = new Instances(dataSet, 0, TESTING_SET_SIZE);
        Instances trainingSet = new Instances(dataSet, TESTING_SET_SIZE, DATASET_SIZE - TESTING_SET_SIZE);
        Classifier classifier = trainWekaClassifier(trainingSet);

        // initialize confusion matrix
        int[][] confMtx = new int[3][];
        for (int i = 0; i < confMtx.length; i++) {
            confMtx[i] = new int[3]; // confusion matrix
            Arrays.fill(confMtx[i], 0);
        }

        // fill the confusion matrix
        try {
            for (int i = 0; i < testingSet.numInstances(); i++) {
                Instance wekaInstance = testingSet.instance(i);
                int targetIndex = (int) Math.round(classifier.classifyInstance(wekaInstance));
                int actualIndex = (int) wekaInstance.value(MAX);
                // building the confusion matrix
                targetIndex = (targetIndex > 0)? 1: (targetIndex < 0)? 0: 2;
                actualIndex = (actualIndex > 0)? 1: (actualIndex < 0)? 0: 2;
                confMtx[actualIndex][targetIndex] += 1;
            }
        } catch (Exception e) {
            logger.warn("Evaluation during testing failed");
            System.exit(-1);
        }

        for (int i = 0; i < confMtx.length; i++)
            System.out.println(Arrays.toString(confMtx[i])); // print the confusion matrix
        double fScore0 = fScore(confMtx[0][0], confMtx[1][0] + confMtx[2][0],
                confMtx[0][1] + confMtx[0][2]);
        double fScore1 = fScore(confMtx[1][1], confMtx[0][1] + confMtx[2][1],
                confMtx[1][0] + confMtx[1][2]);
        double fScore2 = fScore(confMtx[2][2], confMtx[1][2] + confMtx[0][2],
                confMtx[2][1] + confMtx[2][0]);
        double meanfScore = (fScore0 * (confMtx[0][0] + confMtx[0][1] + confMtx[0][2]) +
                fScore1 * (confMtx[1][0] + confMtx[1][1] + confMtx[1][2]) +
                fScore2 * (confMtx[2][0] + confMtx[2][1] + confMtx[2][2])) / testingSet.numInstances();
        System.out.println("F scores: "+ fScore0 + ", " + fScore1 + ", " + fScore2 +
                ", mean:" + meanfScore + "\n");
        return meanfScore;
    }

    /**
     * Calculate the f-score given TP, FP, and FN
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
}