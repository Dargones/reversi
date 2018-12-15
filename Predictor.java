package reversi;

import weka.classifiers.Classifier;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.SerializedInstancesLoader;
import weka.core.converters.SerializedInstancesSaver;

import java.io.File;
import java.util.Arrays;

/**
 * Created by alexanderfedchin on 9/13/18.
 */
public class Predictor {

    public static final double LEARNING_RATE = .03;
    public static final double MOMENTUM = .02;
    public static final String HIDDEN_LAYERS = "a";
    public static final int VALIDATION_SIZE = 11;
    public static final int VALIDATION_THRESHOLD = 3;
    public static final boolean DECAY = false;
    public static final int MAX_LEVEL = 57;

    public static long timeStart = System.currentTimeMillis();
    private static Classifier[] classifiers = new Classifier[Main.MAX];

    static {
        Arrays.fill(classifiers, null);
    }


    public static Classifier getClassifier(int level) {
        if (level > MAX_LEVEL)
            return null;
        if (classifiers[level] == null) {
            classifiers[level] = createClassifier("data_level_" + level, "data_level_" + level + "_test");
            System.out.println("Classifier_" + level + " created");
        }
        return classifiers[level];
    }

    public static void main(String args[]) {
        createNewDataset(15, 22, 500);
    }

    public static void gridSearch(int level) {
        Classifier classifier;
        Instances training_set, testing_set;

        // perform the grid search
        double[] learningRates = {0.1};
        String[] hiddenLayers = {"a"};
        int[] validationSizes = {11};
        int[] validationThresholds = {3};
        boolean[] decays = {false};
        double[] momenta = {0.2};

        training_set = loadInstances("data_level_" + level);
        testing_set = loadInstances("data_level_" + level + "_test");

        System.out.println("Instances loaded");

        double best = -1;
        String bestParams = "";
        for (double learningRate: learningRates)
            for (String layers: hiddenLayers)
                for (int validationSize: validationSizes)
                    for (int validationThreshold: validationThresholds)
                        for (double momentum: momenta)
                            for (boolean decay: decays) {
                                String params = learningRate + " " + layers + " " +
                                        validationSize + " " + validationThreshold + " " +
                                        decay + " " + momentum;
                                System.out.println("Parameters: " + params);
                                classifier = train(learningRate, layers,
                                        validationSize, validationThreshold,
                                        decay, momentum, training_set);
                                double curr = test(classifier, testing_set);
                                if (curr > best) {
                                    best = curr;
                                    bestParams = params;
                                }
                            }

        System.out.println("Best result: " + best + ". Params: " + bestParams);

    }

    public static void createNewDataset(int level, int evaluationLevel, int size) {
        System.out.println(level);
        BoardState state = new BoardState();
        Classifier classifier = null;
        Instances training_set, testing_set;
        System.out.println("Program launched");
        if (evaluationLevel != Main.MAX + 1) {
            classifier = createClassifier("data_level_" + evaluationLevel, "data_level_" + evaluationLevel + "_test");
            System.out.println("Model trained");
        }
        timeStart = System.currentTimeMillis();
        training_set = state.getInstances((byte) level, size / 10 * 9, "data_level_" + level, evaluationLevel, classifier);
        timeStart = System.currentTimeMillis();
        testing_set = state.getInstances((byte) level, size / 10, "data_level_" + level +"_test", evaluationLevel, classifier);
        saveInstances(training_set, "data_level_" + level);
        saveInstances(testing_set, "data_level_" + level + "_test");
        Classifier classifier1 = train(LEARNING_RATE, HIDDEN_LAYERS, VALIDATION_SIZE, VALIDATION_THRESHOLD, DECAY, MOMENTUM, training_set);
        test(classifier1, testing_set);
    }

    public static Classifier createClassifier(String file1, String file2) {
        Instances set = loadInstances(file1);
        Instances additional = loadInstances(file2);
        for (int i = 0; i < additional.numInstances(); i++)
            set.add(additional.instance(i));
        return train(LEARNING_RATE, HIDDEN_LAYERS, VALIDATION_SIZE, VALIDATION_THRESHOLD, DECAY, MOMENTUM, set);
    }

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

    public static double test(Classifier classifier, Instances testing_set) {

        int[][] stats = new int[3][];
        for (int i = 0; i < stats.length; i++) {
            stats[i] = new int[3];
            Arrays.fill(stats[i], 0);
        }

        try {
            for (int i = 0; i < testing_set.numInstances(); i++) {
                Instance wekaInstance = testing_set.instance(i);
                int targetIndex = (int) classifier.classifyInstance(wekaInstance);
                int actualIndex = (int) wekaInstance.value(BoardState.attributes.size() - 1);
                if (BoardState.USE_MINIMAX_FOR_PREDICTING) {
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
                } else if (targetIndex == actualIndex)
                    System.out.println("TODO!");
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        for (int i = 0; i < stats.length; i++)
            System.out.println(Arrays.toString(stats[i]));
        double fScore0 = fScore(stats[0][0], stats[1][0] + stats[2][0], stats[0][1] + stats[0][2]);
        double fScore1 = fScore(stats[1][1], stats[0][1] + stats[2][1], stats[1][0] + stats[1][2]);
        double fScore2 = fScore(stats[2][2], stats[1][2] + stats[0][2], stats[2][1] + stats[2][0]);
        double meanfScore = (fScore0 * (stats[0][0] + stats[0][1] + stats[0][2]) +
                fScore1 * (stats[1][0] + stats[1][1] + stats[1][2]) +
                fScore2 * (stats[2][0] + stats[2][1] + stats[2][2])) / testing_set.numInstances();
        System.out.println(fScore0 + " " + fScore1 + " " + fScore2 + " " + meanfScore + "\n");
        return meanfScore;
    }

    public static double fScore(int tp, int fp, int fn) {
        if (tp == 0)
            return 0;
        double precision = (double) (tp) / (tp + fp);
        double recall = (double) (tp) / (tp + fn);
        return 2 * (recall * precision) / (recall + precision);
    }

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
