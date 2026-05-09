package ceka.common;

import ceka.core.Dataset;
import ceka.core.Example;

public final class NeighborUtils {
    private NeighborUtils() {
    }

    public static double[][] minMax(Dataset dataset) {
        double[][] ranges = new double[dataset.numAttributes()][2];
        for (int a = 0; a < dataset.numAttributes(); a++) {
            ranges[a][0] = Double.POSITIVE_INFINITY;
            ranges[a][1] = Double.NEGATIVE_INFINITY;
            if (a == dataset.classIndex() || !dataset.attribute(a).isNumeric()) {
                continue;
            }
            for (int i = 0; i < dataset.getExampleSize(); i++) {
                Example example = dataset.getExampleByIndex(i);
                if (!example.isMissing(a)) {
                    double value = example.value(a);
                    if (value < ranges[a][0]) {
                        ranges[a][0] = value;
                    }
                    if (value > ranges[a][1]) {
                        ranges[a][1] = value;
                    }
                }
            }
        }
        return ranges;
    }

    public static double heomDistance(Dataset dataset, double[][] ranges, int leftIndex, int rightIndex) {
        Example left = dataset.getExampleByIndex(leftIndex);
        Example right = dataset.getExampleByIndex(rightIndex);
        double distance = 0.0;
        for (int a = 0; a < dataset.numAttributes(); a++) {
            if (a == dataset.classIndex()) {
                continue;
            }
            if (left.isMissing(a) || right.isMissing(a)) {
                distance += 1.0;
            } else if (dataset.attribute(a).isNominal()) {
                if (left.value(a) != right.value(a)) {
                    distance += 1.0;
                }
            } else if (dataset.attribute(a).isNumeric()) {
                double width = ranges[a][1] - ranges[a][0];
                if (width != 0.0 && !Double.isInfinite(width)) {
                    double diff = Math.abs(left.value(a) - right.value(a)) / width;
                    distance += diff * diff;
                }
            }
        }
        return Math.sqrt(distance);
    }

    public static double[] featureVector(Dataset dataset, int index) {
        Example example = dataset.getExampleByIndex(index);
        double[] vector = new double[dataset.numAttributes() - 1];
        int pos = 0;
        for (int a = 0; a < dataset.numAttributes(); a++) {
            if (a == dataset.classIndex()) {
                continue;
            }
            vector[pos++] = example.isMissing(a) ? 0.0 : example.value(a);
        }
        return vector;
    }
}
