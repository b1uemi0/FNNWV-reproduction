package ceka.IWMV;

import java.util.HashMap;

import ceka.core.Dataset;
import ceka.core.Example;
import ceka.core.Label;
import ceka.core.MultiNoisyLabelSet;
import ceka.core.Worker;

public class IWMV {
    public static final String NAME = "IWMV";
    private static final int DEFAULT_MAX_ITERATIONS = 50;

    private int maxIterations = DEFAULT_MAX_ITERATIONS;

    public void setMaxIterations(int maxIterations) {
        this.maxIterations = maxIterations;
    }

    public void doInference(Dataset dataset) {
        int numExamples = dataset.getExampleSize();
        int numWorkers = dataset.getWorkerSize();
        int numClasses = dataset.getCategorySize();

        HashMap<String, Integer> workerIndexById = new HashMap<String, Integer>();
        for (int i = 0; i < numWorkers; i++) {
            Worker worker = dataset.getWorkerByIndex(i);
            workerIndexById.put(worker.getId(), i);
        }

        int[][] labels = new int[numExamples][numWorkers];
        boolean[][] observed = new boolean[numExamples][numWorkers];
        int[] labelCountsByWorker = new int[numWorkers];
        for (int i = 0; i < numExamples; i++) {
            Example example = dataset.getExampleByIndex(i);
            MultiNoisyLabelSet noisyLabelSet = example.getMultipleNoisyLabelSet(0);
            for (int j = 0; j < numWorkers; j++) {
                labels[i][j] = -1;
            }
            for (int k = 0; k < noisyLabelSet.getLabelSetSize(); k++) {
                Label noisyLabel = noisyLabelSet.getLabel(k);
                Integer workerIndex = workerIndexById.get(noisyLabel.getWorkerId());
                if (workerIndex != null) {
                    labels[i][workerIndex] = noisyLabel.getValue();
                    observed[i][workerIndex] = true;
                    labelCountsByWorker[workerIndex]++;
                }
            }
        }

        double[] weights = new double[numWorkers];
        for (int i = 0; i < numWorkers; i++) {
            weights[i] = 1.0;
        }

        int[] predictions = new int[numExamples];
        for (int i = 0; i < numExamples; i++) {
            predictions[i] = -1;
        }

        for (int iteration = 0; iteration < maxIterations; iteration++) {
            int[] nextPredictions = inferLabels(labels, observed, weights, numClasses);
            updateWeights(labels, observed, nextPredictions, labelCountsByWorker, weights, numClasses);
            if (samePredictions(predictions, nextPredictions)) {
                predictions = nextPredictions;
                break;
            }
            predictions = nextPredictions;
        }

        predictions = inferLabels(labels, observed, weights, numClasses);
        for (int i = 0; i < numExamples; i++) {
            Example example = dataset.getExampleByIndex(i);
            Label integratedLabel = new Label(null, String.valueOf(predictions[i]), example.getId(), NAME);
            example.setIntegratedLabel(integratedLabel);
        }
        dataset.assignIntegeratedLabel2WekaInstanceClassValue();
    }

    private static int[] inferLabels(int[][] labels, boolean[][] observed, double[] weights, int numClasses) {
        int[] predictions = new int[labels.length];
        for (int i = 0; i < labels.length; i++) {
            double[] scores = new double[numClasses];
            for (int worker = 0; worker < labels[i].length; worker++) {
                if (observed[i][worker]) {
                    scores[labels[i][worker]] += weights[worker];
                }
            }
            predictions[i] = maxIndex(scores);
        }
        return predictions;
    }

    private static void updateWeights(
            int[][] labels,
            boolean[][] observed,
            int[] predictions,
            int[] labelCountsByWorker,
            double[] weights,
            int numClasses) {
        for (int worker = 0; worker < weights.length; worker++) {
            if (labelCountsByWorker[worker] == 0) {
                weights[worker] = 0.0;
                continue;
            }
            int correct = 0;
            for (int example = 0; example < labels.length; example++) {
                if (observed[example][worker] && labels[example][worker] == predictions[example]) {
                    correct++;
                }
            }
            double reliability = (double) correct / (double) labelCountsByWorker[worker];
            weights[worker] = numClasses * reliability - 1.0;
        }
    }

    private static boolean samePredictions(int[] left, int[] right) {
        if (left.length != right.length) {
            return false;
        }
        for (int i = 0; i < left.length; i++) {
            if (left[i] != right[i]) {
                return false;
            }
        }
        return true;
    }

    private static int maxIndex(double[] values) {
        int max = 0;
        for (int i = 1; i < values.length; i++) {
            if (values[i] > values[max]) {
                max = i;
            }
        }
        return max;
    }
}
