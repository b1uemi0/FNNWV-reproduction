package ceka.FNNWV;

import ceka.LAWMV.LAWMV;
import ceka.consensus.MajorityVote;
import ceka.core.Dataset;
import ceka.core.Example;
import ceka.core.Label;
import ceka.core.MultiNoisyLabelSet;
import weka.core.Utils;

public class FNNWVVariant {
    public enum Mode {
        MV_ONLY,
        NEAREST_ONLY,
        FARTHEST_ONLY
    }

    private final Mode mode;

    public FNNWVVariant(Mode mode) {
        this.mode = mode;
    }

    public void doInference(Dataset dataset) throws Exception {
        if (mode == Mode.MV_ONLY) {
            new MajorityVote().doInference(dataset);
        } else if (mode == Mode.NEAREST_ONLY) {
            new LAWMV().doInference(dataset);
        } else {
            doFarthestOnly(dataset);
        }
    }

    private static void doFarthestOnly(Dataset dataset) throws Exception {
        int numExample = dataset.getExampleSize();
        int knn = (int) (numExample / dataset.getCategorySize() * 0.5);
        double[] attMax = new double[dataset.numAttributes()];
        double[] attMin = new double[dataset.numAttributes()];
        for (int i = 0; i < dataset.numAttributes(); i++) {
            if (i != dataset.classIndex() && dataset.attribute(i).isNumeric()) {
                double[] values = new double[numExample];
                for (int j = 0; j < numExample; j++) {
                    values[j] = dataset.getExampleByIndex(j).value(i);
                }
                attMin[i] = values[Utils.minIndex(values)];
                attMax[i] = values[Utils.maxIndex(values)];
            }
        }

        for (int i = 0; i < numExample; i++) {
            double[] distance = new double[numExample];
            double[] weight = new double[numExample];
            double[] weightSum = new double[numExample];
            Example example = dataset.getExampleByIndex(i);
            MultiNoisyLabelSet labels = example.getMultipleNoisyLabelSet(0);
            for (int j = 0; j < numExample; j++) {
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
                for (int k = 0; k < dataset.numAttributes(); k++) {
                    if (k == dataset.classIndex()) {
                        continue;
                    }
                    if (example.isMissing(k) || other.isMissing(k)) {
                        distance[j] += 1.0;
                    } else if (dataset.attribute(k).isNominal()) {
                        if (example.value(k) != other.value(k)) {
                            distance[j] += 1.0;
                        }
                    } else if (dataset.attribute(k).isNumeric()) {
                        double width = attMax[k] - attMin[k];
                        if (width != 0.0) {
                            distance[j] += Math.pow(Math.abs(example.value(k) - other.value(k)) / width, 2);
                        }
                    }
                }
                distance[j] = Math.sqrt(distance[j]);
            }

            int[] sorted = Utils.sort(distance);
            double[] classCounts = new double[dataset.numClasses()];
            double farthestDistance = distance[sorted[sorted.length - 1]];
            for (int j = 0; j < knn; j++) {
                int idx = sorted[sorted.length - 1 - j];
                MultiNoisyLabelSet neighborLabels = dataset.getExampleByIndex(idx).getMultipleNoisyLabelSet(0);
                double tempWeight = (1.0 - weight[idx]) + (farthestDistance == 0.0 ? 0.0 : distance[idx] / farthestDistance);
                tempWeight *= 0.5;
                for (int k = 0; k < neighborLabels.getLabelSetSize(); k++) {
                    classCounts[neighborLabels.getLabel(k).getValue()] -= tempWeight;
                }
            }

            int maxIndex = 0;
            double maxValue = -10000.0;
            for (int j = 0; j < classCounts.length; j++) {
                if (classCounts[j] > maxValue) {
                    maxValue = classCounts[j];
                    maxIndex = j;
                }
            }
            example.setIntegratedLabel(new Label(null, String.valueOf(maxIndex), example.getId(), "FNNWV-F"));
        }
        dataset.assignIntegeratedLabel2WekaInstanceClassValue();
    }
}
