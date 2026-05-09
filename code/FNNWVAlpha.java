package ceka.FNNWV;

import java.util.Arrays;

import ceka.core.Dataset;
import ceka.core.Example;
import ceka.core.Label;
import ceka.core.MultiNoisyLabelSet;
import weka.core.Utils;

public class FNNWVAlpha {
    public enum ThresholdMode {
        FIXED,
        QUANTILE
    }

    public static class Stats {
        public double threshold;
        public int triggered;
        public int total;

        public double triggerRate() {
            return total == 0 ? 0.0 : (double) triggered / (double) total;
        }
    }

    private final ThresholdMode mode;
    private final double parameter;
    private final String name;
    private Stats lastStats = new Stats();

    public FNNWVAlpha(double alpha) {
        this(ThresholdMode.FIXED, alpha, "FNNWV-alpha-" + alpha);
    }

    public FNNWVAlpha(ThresholdMode mode, double parameter, String name) {
        this.mode = mode;
        this.parameter = parameter;
        this.name = name;
    }

    public Stats getLastStats() {
        return lastStats;
    }

    public void doInference(Dataset dataset) throws Exception {
        int numExample = dataset.getExampleSize();
        int knn = (int) (numExample / dataset.getCategorySize() * 0.5);
        if (knn < 1) {
            throw new IllegalArgumentException("Dataset is too small for FNNWV.");
        }

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

        ExampleContext[] contexts = new ExampleContext[numExample];
        double[] margins = new double[numExample];
        for (int i = 0; i < numExample; i++) {
            contexts[i] = buildContext(dataset, i, knn, attMin, attMax);
            margins[i] = contexts[i].margin;
        }

        double threshold = chooseThreshold(margins);
        int triggered = 0;
        for (int i = 0; i < numExample; i++) {
            ExampleContext context = contexts[i];
            double[] classCounts = Arrays.copyOf(context.nearestClassCounts, context.nearestClassCounts.length);
            if (context.margin < threshold) {
                triggered++;
                applyFarthestCorrection(dataset, context, classCounts, knn);
            }
            int maxIndex = maxIndex(classCounts);
            Example example = dataset.getExampleByIndex(i);
            example.setIntegratedLabel(new Label(null, String.valueOf(maxIndex), example.getId(), name));
        }

        Stats stats = new Stats();
        stats.threshold = threshold;
        stats.triggered = triggered;
        stats.total = numExample;
        lastStats = stats;
        dataset.assignIntegeratedLabel2WekaInstanceClassValue();
    }

    private double chooseThreshold(double[] margins) {
        if (mode == ThresholdMode.FIXED) {
            return parameter;
        }
        double q = Math.max(0.0, Math.min(1.0, parameter));
        double[] copy = Arrays.copyOf(margins, margins.length);
        Arrays.sort(copy);
        int index = (int) Math.floor(q * (copy.length - 1));
        return copy[index];
    }

    private static ExampleContext buildContext(Dataset dataset, int targetIndex, int knn,
            double[] attMin, double[] attMax) {
        int numExample = dataset.getExampleSize();
        double[] distance = new double[numExample];
        double[] weight = new double[numExample];
        double[] weightSum = new double[numExample];
        Example target = dataset.getExampleByIndex(targetIndex);
        MultiNoisyLabelSet targetLabels = target.getMultipleNoisyLabelSet(0);

        for (int j = 0; j < numExample; j++) {
            Example other = dataset.getExampleByIndex(j);
            for (int k = 0; k < targetLabels.getLabelSetSize(); k++) {
                Label label = other.getNoisyLabelByWorkerId(targetLabels.getLabel(k).getWorkerId());
                if (label != null) {
                    weightSum[j]++;
                    if (label.getValue() == targetLabels.getLabel(k).getValue()) {
                        weight[j] += 1.0;
                    }
                }
            }
            weight[j] = weightSum[j] == 0.0 ? 0.0 : weight[j] / weightSum[j];
            for (int k = 0; k < dataset.numAttributes(); k++) {
                if (k == dataset.classIndex()) {
                    continue;
                }
                if (target.isMissing(k) || other.isMissing(k)) {
                    distance[j] += 1.0;
                } else if (dataset.attribute(k).isNominal()) {
                    if (target.value(k) != other.value(k)) {
                        distance[j] += 1.0;
                    }
                } else if (dataset.attribute(k).isNumeric()) {
                    double width = attMax[k] - attMin[k];
                    if (width != 0.0) {
                        distance[j] += Math.pow(Math.abs(target.value(k) - other.value(k)) / width, 2);
                    }
                }
            }
            distance[j] = Math.sqrt(distance[j]);
        }

        int[] sorted = Utils.sort(distance);
        double[] nearestClassCounts = new double[dataset.numClasses()];
        double kthDistance = distance[sorted[knn - 1]];
        for (int j = 0; j < knn; j++) {
            int idx = sorted[j];
            MultiNoisyLabelSet labels = dataset.getExampleByIndex(idx).getMultipleNoisyLabelSet(0);
            double distanceWeight = kthDistance == 0.0 ? 1.0 : 1.0 - (distance[idx] / kthDistance);
            double voteWeight = 0.5 * (weight[idx] + distanceWeight);
            for (int k = 0; k < labels.getLabelSetSize(); k++) {
                nearestClassCounts[labels.getLabel(k).getValue()] += voteWeight;
            }
        }

        ExampleContext context = new ExampleContext();
        context.distance = distance;
        context.weight = weight;
        context.sorted = sorted;
        context.nearestClassCounts = nearestClassCounts;
        context.margin = margin(nearestClassCounts);
        return context;
    }

    private static void applyFarthestCorrection(Dataset dataset, ExampleContext context,
            double[] classCounts, int knn) {
        int numExample = dataset.getExampleSize();
        double farthestDistance = context.distance[context.sorted[numExample - 1]];
        for (int j = 0; j < knn; j++) {
            int idx = context.sorted[numExample - 1 - j];
            MultiNoisyLabelSet labels = dataset.getExampleByIndex(idx).getMultipleNoisyLabelSet(0);
            double distanceWeight = farthestDistance == 0.0 ? 0.0 : context.distance[idx] / farthestDistance;
            double voteWeight = 0.5 * ((1.0 - context.weight[idx]) + distanceWeight);
            for (int k = 0; k < labels.getLabelSetSize(); k++) {
                classCounts[labels.getLabel(k).getValue()] -= voteWeight;
            }
        }
    }

    private static double margin(double[] classCounts) {
        double sum = Utils.sum(classCounts);
        if (sum == 0.0) {
            return 0.0;
        }
        double top1 = -Double.MAX_VALUE;
        double top2 = -Double.MAX_VALUE;
        for (double value : classCounts) {
            if (value > top1) {
                top2 = top1;
                top1 = value;
            } else if (value > top2) {
                top2 = value;
            }
        }
        if (top2 == -Double.MAX_VALUE) {
            top2 = 0.0;
        }
        return Math.abs(top1 - top2) / sum;
    }

    private static int maxIndex(double[] classCounts) {
        int maxIndex = 0;
        double maxValue = -Double.MAX_VALUE;
        for (int i = 0; i < classCounts.length; i++) {
            if (classCounts[i] > maxValue) {
                maxValue = classCounts[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    private static class ExampleContext {
        double[] distance;
        double[] weight;
        int[] sorted;
        double[] nearestClassCounts;
        double margin;
    }
}
