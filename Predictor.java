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
    public static final String DEFAULT_TRAINING = "data_level_25";
    public static final String DEFAULT_TESTING = "data_level_25_test";


    public static void main(String args[]) {

        BoardState state = new BoardState();
        Classifier classifier;
        Instances training_set, testing_set;

        /* classifier = getClassifier(DEFAULT_TRAINING, DEFAULT_TESTING);
        training_set = state.getInstances((byte) 20, 4500, "data_level_20", 25, classifier);
        testing_set = state.getInstances((byte) 20, 1000, "data_level_20_test", 25, classifier);
        saveInstances(training_set, "data_level_20");
        saveInstances(testing_set, "data_level_20_test");
        Classifier classifier1 = train(LEARNING_RATE, HIDDEN_LAYERS, VALIDATION_SIZE, VALIDATION_THRESHOLD, DECAY, MOMENTUM, training_set);
        test(classifier1, testing_set); */


        // perform the grid search
        double[] learningRates = {.01, .1, 1, 0.5, 0.2};
        String[] hiddenLayers = {"a", "i", "i,a", "a,a"};
        int[] validationSizes = {11};
        int[] validationThresholds = {1, 5, 10};
        boolean[] decays = {false, true};
        double[] momenta = {0.1, 0.2, 0.8};

        training_set = loadInstances(DEFAULT_TRAINING);
        testing_set = loadInstances(DEFAULT_TESTING);
        if ((training_set == null) || (testing_set == null)) {
            training_set = state.getInstances((byte) 25, 900, DEFAULT_TRAINING, Main.MAX + 1, null);
            testing_set = state.getInstances((byte) 25, 100, DEFAULT_TESTING, Main.MAX + 1, null);
            saveInstances(training_set, DEFAULT_TRAINING);
            saveInstances(testing_set, DEFAULT_TESTING);
        }

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

    public static Classifier getClassifier(String file1, String file2) {
        Instances set = loadInstances(file1);
        Instances additional = loadInstances(file2);
        for (int i = 0; i < additional.numInstances(); i++) {
            set.add(additional.instance(i));
        }
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
