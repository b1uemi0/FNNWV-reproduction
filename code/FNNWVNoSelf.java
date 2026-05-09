package ceka.FNNWV;

import ceka.common.NeighborUtils;
import ceka.core.Dataset;
import ceka.core.Example;
import ceka.core.Label;
import ceka.core.MultiNoisyLabelSet;
import weka.core.Utils;

public class FNNWVNoSelf {
    public static final String NAME = "FNNWVNoSelf";

    public void doInference(Dataset dataset) throws Exception {
        int numExample = dataset.getExampleSize();
        int knn = (int) (numExample / dataset.getCategorySize() * 0.5);
        double[][] ranges = NeighborUtils.minMax(dataset);
        for (int i = 0; i < numExample; i++) {
            double[] distance = new double[numExample];
            double[] weight = new double[numExample];
            double[] weightSum = new double[numExample];
            Example example = dataset.getExampleByIndex(i);
            MultiNoisyLabelSet labels = example.getMultipleNoisyLabelSet(0);
            for (int j = 0; j < numExample; j++) {
                distance[j] = i == j ? Double.POSITIVE_INFINITY : NeighborUtils.heomDistance(dataset, ranges, i, j);
                Example other = dataset.getExampleByIndex(j);
                for (int k = 0; k < labels.getLabelSetSize(); k++) {
                    Label label = other.getNoisyLabelByWorkerId(labels.getLabel(k).getWorkerId());
                    if (label != null) {
                        weightSum[j]++;
                        if (label.getValue() == labels.getLabel(k).getValue()) {
                            weight[j] += 1.0;
                        }
                    }
                }
                weight[j] = weightSum[j] == 0.0 ? 0.0 : weight[j] / weightSum[j];
            }
            int[] sorted = Utils.sort(distance);
            double[] classCounts = new double[dataset.numClasses()];
            double kthDistance = distance[sorted[knn - 1]];
            for (int j = 0; j < knn; j++) {
                int idx = sorted[j];
                MultiNoisyLabelSet neighborLabels = dataset.getExampleByIndex(idx).getMultipleNoisyLabelSet(0);
                double voteWeight = 0.5 * (weight[idx] + (1.0 - distance[idx] / kthDistance));
                for (int k = 0; k < neighborLabels.getLabelSetSize(); k++) {
                    classCounts[neighborLabels.getLabel(k).getValue()] += voteWeight;
                }
            }
            double sum = Utils.sum(classCounts);
            double nd = sum == 0.0 ? 0.0 : Math.abs(classCounts[0] - classCounts[1]) / sum;
            if (nd < 0.1) {
                double[] farDistances = new double[numExample];
                for (int j = 0; j < numExample; j++) {
                    farDistances[j] = i == j ? Double.NEGATIVE_INFINITY : NeighborUtils.heomDistance(dataset, ranges, i, j);
                }
                int[] farSorted = Utils.sort(farDistances);
                double farthestDistance = farDistances[farSorted[farSorted.length - 1]];
                for (int j = 0; j < knn; j++) {
                    int idx = farSorted[farSorted.length - 1 - j];
                    MultiNoisyLabelSet neighborLabels = dataset.getExampleByIndex(idx).getMultipleNoisyLabelSet(0);
                    double voteWeight = 0.5 * ((1.0 - weight[idx]) + farDistances[idx] / farthestDistance);
                    for (int k = 0; k < neighborLabels.getLabelSetSize(); k++) {
                        classCounts[neighborLabels.getLabel(k).getValue()] -= voteWeight;
                    }
                }
            }
            int best = 0;
            for (int c = 1; c < classCounts.length; c++) {
                if (classCounts[c] > classCounts[best]) {
                    best = c;
                }
            }
            example.setIntegratedLabel(new Label(null, String.valueOf(best), example.getId(), NAME));
        }
        dataset.assignIntegeratedLabel2WekaInstanceClassValue();
    }
}
