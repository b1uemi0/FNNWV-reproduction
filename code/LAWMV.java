package ceka.LAWMV;

import java.util.HashMap;

import ceka.common.NeighborUtils;
import ceka.core.Dataset;
import ceka.core.Example;
import ceka.core.Label;
import ceka.core.MultiNoisyLabelSet;
import weka.core.Utils;

public class LAWMV {
    public static final String NAME = "LAWMV";
    private double b = 0.5;

    public void setB(double b) {
        this.b = b;
    }

    public void doInference(Dataset dataset) throws Exception {
        int numExamples = dataset.getExampleSize();
        int numClasses = dataset.getCategorySize();
        int kNeighbors = Math.max(1, (int) (b * numExamples / numClasses));
        double[][] ranges = NeighborUtils.minMax(dataset);

        for (int i = 0; i < numExamples; i++) {
            double[] distances = new double[numExamples];
            for (int j = 0; j < numExamples; j++) {
                distances[j] = NeighborUtils.heomDistance(dataset, ranges, i, j);
            }
            int[] sorted = Utils.sort(distances);
            double maxNeighborDistance = distances[sorted[kNeighbors - 1]];
            double[] classScores = new double[numClasses];

            for (int k = 0; k < kNeighbors; k++) {
                int neighborIndex = sorted[k];
                double w1 = maxNeighborDistance == 0.0
                        ? 1.0
                        : 1.0 - distances[neighborIndex] / maxNeighborDistance;
                double w2 = labelSimilarity(dataset, i, neighborIndex);
                double weight = (w1 + w2) * 0.5;
                MultiNoisyLabelSet labels = dataset.getExampleByIndex(neighborIndex).getMultipleNoisyLabelSet(0);
                for (int l = 0; l < labels.getLabelSetSize(); l++) {
                    classScores[labels.getLabel(l).getValue()] += weight;
                }
            }

            int integrated = maxIndex(classScores);
            Example example = dataset.getExampleByIndex(i);
            example.setIntegratedLabel(new Label(null, String.valueOf(integrated), example.getId(), NAME));
        }
        dataset.assignIntegeratedLabel2WekaInstanceClassValue();
    }

    private static double labelSimilarity(Dataset dataset, int leftIndex, int rightIndex) {
        MultiNoisyLabelSet left = dataset.getExampleByIndex(leftIndex).getMultipleNoisyLabelSet(0);
        MultiNoisyLabelSet right = dataset.getExampleByIndex(rightIndex).getMultipleNoisyLabelSet(0);
        HashMap<String, Integer> rightLabels = new HashMap<String, Integer>();
        for (int i = 0; i < right.getLabelSetSize(); i++) {
            Label label = right.getLabel(i);
            rightLabels.put(label.getWorkerId(), label.getValue());
        }
        int shared = 0;
        int same = 0;
        for (int i = 0; i < left.getLabelSetSize(); i++) {
            Label label = left.getLabel(i);
            Integer other = rightLabels.get(label.getWorkerId());
            if (other != null) {
                shared++;
                if (other.intValue() == label.getValue()) {
                    same++;
                }
            }
        }
        return shared == 0 ? 0.0 : (double) same / (double) shared;
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
