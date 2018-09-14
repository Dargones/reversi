package reversi;

import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.SerializedInstancesLoader;
import weka.core.converters.SerializedInstancesSaver;

import java.io.File;

/**
 * Created by alexanderfedchin on 9/13/18.
 */
public class Predictor {


    public static void main(String args[]) {
        // perform the grid search
        double[] learningRates = {.01, .1, 1, 0.5, 0.2};
        String[] hiddenLayers = {"a", "i,a", "a,a"};
        int[] validationSizes = {10, 20};
        int[] validationThresholds = {1, 5, 10};
        boolean[] decays = {false};
        double[] momenta = {0.1, 0.2, 0.5, 0.8};

        BoardState state = new BoardState();
        Instances training_set = loadInstances("data_training2");
        Instances testing_set = loadInstances("data_testing2");
        if ((training_set == null) || (testing_set == null)) {
            training_set = state.getInstances((byte) 25, 9000, "training");
            testing_set = state.getInstances((byte) 25, 1000, "testing");
            saveInstances(training_set, "data_training2");
            saveInstances(testing_set, "data_testing2");
        }

        System.out.println("Instances loaded");

        int best = -1;
        String bestParams = "";
        for (double learningRate: learningRates)
            for (String layers: hiddenLayers)
                for (int validationSize: validationSizes)
                    for (int validationThreshold: validationThresholds)
                        for (double momentum: momenta)
                            for (boolean decay: decays) {
                                int curr = test(learningRate, layers,
                                        validationSize, validationThreshold,
                                        decay, momentum, training_set, testing_set);
                                String params = learningRate + " " + layers + " " +
                                        validationSize + " " + validationThreshold + " " +
                                        decay + " " + momentum;
                                System.out.println(params + "\n" + curr + "\n");
                                if (curr > best)
                                    best = curr;
                                bestParams = params;
                            }

        System.out.println("Best result: " + best + ". Params: " + bestParams);

    }

    public static int test(double learningRate, String hiddenLayers,
                           int validationSize, int validationThreshold,
                           boolean decay, double momentum,
                           Instances training_set, Instances testing_set) {

        MultilayerPerceptron classifier = new MultilayerPerceptron();
        classifier.setValidationSetSize(validationSize);
        classifier.setDecay(decay);
        classifier.setValidationThreshold(validationThreshold);
        classifier.setLearningRate(learningRate);
        classifier.setHiddenLayers(hiddenLayers);
        classifier.setMomentum(momentum);

        // System.out.println("Network built");
        
        int correct = 0;
        try {
            classifier.buildClassifier(training_set);

            // System.out.println("Network trained");

            for (int i = 0; i < testing_set.numInstances(); i++) {
                Instance wekaInstance = testing_set.instance(i);
                int targetIndex = (int) classifier.classifyInstance(wekaInstance);
                if (targetIndex == wekaInstance.value(BoardState.attributes.size() - 1))
                    correct += 1;
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        return correct;
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
