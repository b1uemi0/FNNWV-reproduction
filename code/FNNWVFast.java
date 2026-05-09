package ceka.FNNWV;

import java.util.Arrays;
import java.util.BitSet;
import java.util.HashMap;

import ceka.core.Dataset;
import ceka.core.Example;
import ceka.core.Label;
import ceka.core.MultiNoisyLabelSet;
import ceka.core.Worker;

public class FNNWVFast {
    public static final String NAME = "FNNWVFast";

    public void doInference(Dataset dataset) throws Exception {
        if (!isBinaryNominalDataset(dataset)) {
            new FNNWV().doInference(dataset);
            return;
        }

        int numExample = dataset.getExampleSize();
        int numClasses = dataset.numClasses();
        int knn = (int) (numExample / dataset.getCategorySize() * 0.5);
        if (knn < 1) {
            throw new IllegalArgumentException("Dataset is too small for FNNWV.");
        }

        BitSet[] featureBits = buildFeatureBits(dataset);
        WorkerIndex workerIndex = buildWorkerIndex(dataset);
        int[][] noisyValues = buildNoisyValueMatrix(dataset, workerIndex.workerIdToIndex);
        int[][] labelWorkerIndexes = new int[numExample][];
        int[][] labelValues = new int[numExample][];
        for (int i = 0; i < numExample; i++) {
            MultiNoisyLabelSet labels = dataset.getExampleByIndex(i).getMultipleNoisyLabelSet(0);
            labelWorkerIndexes[i] = new int[labels.getLabelSetSize()];
            labelValues[i] = new int[labels.getLabelSetSize()];
            for (int k = 0; k < labels.getLabelSetSize(); k++) {
                Label label = labels.getLabel(k);
                labelWorkerIndexes[i][k] = workerIndex.workerIdToIndex.get(label.getWorkerId());
                labelValues[i][k] = label.getValue();
            }
        }

        long start = System.currentTimeMillis();
        for (int i = 0; i < numExample; i++) {
            if (i > 0 && i % 100 == 0) {
                long elapsed = System.currentTimeMillis() - start;
                System.out.println("FNNWVFast progress: " + i + "/" + numExample + " examples, elapsed_ms=" + elapsed);
            }
            double[] distances = new double[numExample];
            double[] weights = new double[numExample];

            for (int j = 0; j < numExample; j++) {
                weights[j] = agreementWeight(labelWorkerIndexes[i], labelValues[i], noisyValues[j]);
                distances[j] = Math.sqrt(hammingDistance(featureBits[i], featureBits[j]));
            }

            Integer[] sorted = new Integer[numExample];
            for (int j = 0; j < numExample; j++) {
                sorted[j] = j;
            }
            Arrays.sort(sorted, (a, b) -> Double.compare(distances[a], distances[b]));

            double[] classCounts = new double[numClasses];
            double kthDistance = distances[sorted[knn - 1]];
            for (int j = 0; j < knn; j++) {
                int idx = sorted[j];
                double distanceWeight = kthDistance == 0.0 ? 1.0 : 1.0 - (distances[idx] / kthDistance);
                double voteWeight = 0.5 * (weights[idx] + distanceWeight);
                for (int value : labelValues[idx]) {
                    classCounts[value] += voteWeight;
                }
            }

            double sum = 0.0;
            for (double value : classCounts) {
                sum += value;
            }
            double margin = sum == 0.0 ? 0.0 : Math.abs(classCounts[0] - classCounts[1]) / sum;
            if (margin < 0.1) {
                double farthestDistance = distances[sorted[numExample - 1]];
                for (int j = 0; j < knn; j++) {
                    int idx = sorted[numExample - 1 - j];
                    double distanceWeight = farthestDistance == 0.0 ? 0.0 : distances[idx] / farthestDistance;
                    double voteWeight = 0.5 * ((1.0 - weights[idx]) + distanceWeight);
                    for (int value : labelValues[idx]) {
                        classCounts[value] -= voteWeight;
                    }
                }
            }

            int maxIndex = 0;
            double maxValue = -10000.0;
            for (int j = 0; j < numClasses; j++) {
                if (classCounts[j] > maxValue) {
                    maxValue = classCounts[j];
                    maxIndex = j;
                }
            }

            Example example = dataset.getExampleByIndex(i);
            Label label = new Label(null, String.valueOf(maxIndex), example.getId(), NAME);
            example.setIntegratedLabel(label);
        }
        dataset.assignIntegeratedLabel2WekaInstanceClassValue();
    }

    private static boolean isBinaryNominalDataset(Dataset dataset) {
        for (int i = 0; i < dataset.numAttributes(); i++) {
            if (i == dataset.classIndex()) {
                continue;
            }
            if (!dataset.attribute(i).isNominal() || dataset.attribute(i).numValues() > 2) {
                return false;
            }
        }
        return true;
    }

    private static BitSet[] buildFeatureBits(Dataset dataset) {
        int numExample = dataset.getExampleSize();
        BitSet[] bits = new BitSet[numExample];
        for (int i = 0; i < numExample; i++) {
            Example example = dataset.getExampleByIndex(i);
            BitSet bitSet = new BitSet(dataset.numAttributes());
            for (int a = 0; a < dataset.numAttributes(); a++) {
                if (a != dataset.classIndex() && !example.isMissing(a) && example.value(a) != 0.0) {
                    bitSet.set(a);
                }
            }
            bits[i] = bitSet;
        }
        return bits;
    }

    private static int hammingDistance(BitSet left, BitSet right) {
        BitSet xored = (BitSet) left.clone();
        xored.xor(right);
        return xored.cardinality();
    }

    private static double agreementWeight(int[] workerIndexes, int[] values, int[] otherNoisyValues) {
        int shared = 0;
        int agree = 0;
        for (int k = 0; k < workerIndexes.length; k++) {
            int otherValue = otherNoisyValues[workerIndexes[k]];
            if (otherValue >= 0) {
                shared++;
                if (otherValue == values[k]) {
                    agree++;
                }
            }
        }
        return shared == 0 ? 0.0 : (double) agree / (double) shared;
    }

    private static int[][] buildNoisyValueMatrix(Dataset dataset, HashMap<String, Integer> workerIdToIndex) {
        int[][] values = new int[dataset.getExampleSize()][workerIdToIndex.size()];
        for (int i = 0; i < values.length; i++) {
            Arrays.fill(values[i], -1);
            MultiNoisyLabelSet labels = dataset.getExampleByIndex(i).getMultipleNoisyLabelSet(0);
            for (int k = 0; k < labels.getLabelSetSize(); k++) {
                Label label = labels.getLabel(k);
                values[i][workerIdToIndex.get(label.getWorkerId())] = label.getValue();
            }
        }
        return values;
    }

    private static WorkerIndex buildWorkerIndex(Dataset dataset) {
        HashMap<String, Integer> workerIdToIndex = new HashMap<String, Integer>();
        for (int i = 0; i < dataset.getWorkerSize(); i++) {
            Worker worker = dataset.getWorkerByIndex(i);
            workerIdToIndex.put(worker.getId(), i);
        }
        return new WorkerIndex(workerIdToIndex);
    }

    private static class WorkerIndex {
        final HashMap<String, Integer> workerIdToIndex;

        WorkerIndex(HashMap<String, Integer> workerIdToIndex) {
            this.workerIdToIndex = workerIdToIndex;
        }
    }
}
