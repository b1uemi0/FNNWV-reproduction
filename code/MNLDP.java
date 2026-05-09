package ceka.MNLDP;

import ceka.common.NeighborUtils;
import ceka.core.Dataset;
import ceka.core.Example;
import ceka.core.Label;
import ceka.core.MultiNoisyLabelSet;
import weka.core.Utils;

public class MNLDP {
    public static final String NAME = "MNLDP";
    private int kNeighbors = 5;
    private double eta = 0.5;
    private int iterations = 20;

    public void setMyQP(Object ignored) {
        // Compatibility hook for the original experiment code. This implementation
        // solves the small local LLE systems directly and does not need MyQP.
    }

    public void doInference(Dataset dataset) throws Exception {
        int numExamples = dataset.getExampleSize();
        int numClasses = dataset.getCategorySize();
        double[][] ranges = NeighborUtils.minMax(dataset);

        int[][] neighbors = new int[numExamples][kNeighbors];
        double[][] neighborWeights = new double[numExamples][kNeighbors];
        for (int i = 0; i < numExamples; i++) {
            double[] distances = new double[numExamples];
            for (int j = 0; j < numExamples; j++) {
                distances[j] = i == j ? Double.POSITIVE_INFINITY : NeighborUtils.heomDistance(dataset, ranges, i, j);
            }
            int[] sorted = Utils.sort(distances);
            for (int k = 0; k < kNeighbors; k++) {
                neighbors[i][k] = sorted[k];
            }
            neighborWeights[i] = localLinearWeights(dataset, i, neighbors[i]);
        }

        double[][] initial = initialDistributions(dataset, numClasses);
        double[] alpha = alphaValues(dataset, numClasses);
        double[][] current = copy(initial);

        for (int t = 0; t < iterations; t++) {
            double[][] next = new double[numExamples][numClasses];
            for (int i = 0; i < numExamples; i++) {
                for (int c = 0; c < numClasses; c++) {
                    double propagated = 0.0;
                    for (int k = 0; k < kNeighbors; k++) {
                        propagated += neighborWeights[i][k] * current[neighbors[i][k]][c];
                    }
                    next[i][c] = alpha[i] * propagated + (1.0 - alpha[i]) * initial[i][c];
                }
                normalize(next[i]);
            }
            current = next;
        }

        for (int i = 0; i < numExamples; i++) {
            int integrated = maxIndex(current[i]);
            Example example = dataset.getExampleByIndex(i);
            example.setIntegratedLabel(new Label(null, String.valueOf(integrated), example.getId(), NAME));
        }
        dataset.assignIntegeratedLabel2WekaInstanceClassValue();
    }

    private static double[][] initialDistributions(Dataset dataset, int numClasses) {
        double[][] distributions = new double[dataset.getExampleSize()][numClasses];
        for (int i = 0; i < dataset.getExampleSize(); i++) {
            MultiNoisyLabelSet labels = dataset.getExampleByIndex(i).getMultipleNoisyLabelSet(0);
            for (int j = 0; j < labels.getLabelSetSize(); j++) {
                distributions[i][labels.getLabel(j).getValue()] += 1.0;
            }
            normalize(distributions[i]);
        }
        return distributions;
    }

    private double[] alphaValues(Dataset dataset, int numClasses) {
        double[] alpha = new double[dataset.getExampleSize()];
        for (int i = 0; i < dataset.getExampleSize(); i++) {
            MultiNoisyLabelSet labels = dataset.getExampleByIndex(i).getMultipleNoisyLabelSet(0);
            int total = labels.getLabelSetSize();
            double sumUncertainty = 0.0;
            for (int c = 0; c < numClasses; c++) {
                int positive = 0;
                for (int j = 0; j < total; j++) {
                    if (labels.getLabel(j).getValue() == c) {
                        positive++;
                    }
                }
                int negative = total - positive;
                double id = binomialUpperTail(positive + 1, negative + 1, 0.5);
                sumUncertainty += Math.min(id, 1.0 - id);
            }
            alpha[i] = Math.min(1.0, Math.max(0.0, sumUncertainty / numClasses + eta));
        }
        return alpha;
    }

    private static double binomialUpperTail(int positive, int negative, double d) {
        int n = positive + negative;
        double sum = 0.0;
        for (int j = positive; j <= n; j++) {
            sum += combinationProbability(n, j, d);
        }
        return sum;
    }

    private static double combinationProbability(int n, int k, double p) {
        double result = 1.0;
        for (int i = 1; i <= k; i++) {
            result *= (double) (n - k + i) / (double) i;
        }
        return result * Math.pow(p, k) * Math.pow(1.0 - p, n - k);
    }

    private static double[] localLinearWeights(Dataset dataset, int index, int[] neighbors) {
        double[] xi = NeighborUtils.featureVector(dataset, index);
        double[][] z = new double[neighbors.length][xi.length];
        for (int i = 0; i < neighbors.length; i++) {
            double[] xj = NeighborUtils.featureVector(dataset, neighbors[i]);
            for (int a = 0; a < xi.length; a++) {
                z[i][a] = xi[a] - xj[a];
            }
        }

        double[][] gram = new double[neighbors.length][neighbors.length];
        double trace = 0.0;
        for (int i = 0; i < neighbors.length; i++) {
            for (int j = 0; j < neighbors.length; j++) {
                double value = 0.0;
                for (int a = 0; a < xi.length; a++) {
                    value += z[i][a] * z[j][a];
                }
                gram[i][j] = value;
            }
            trace += gram[i][i];
        }
        double regularizer = trace == 0.0 ? 1e-3 : trace * 1e-3;
        for (int i = 0; i < neighbors.length; i++) {
            gram[i][i] += regularizer;
        }

        double[] ones = new double[neighbors.length];
        for (int i = 0; i < ones.length; i++) {
            ones[i] = 1.0;
        }
        double[] weights = solveLinearSystem(gram, ones);
        boolean hasPositive = false;
        for (int i = 0; i < weights.length; i++) {
            if (Double.isNaN(weights[i]) || Double.isInfinite(weights[i]) || weights[i] < 0.0) {
                weights[i] = 0.0;
            }
            if (weights[i] > 0.0) {
                hasPositive = true;
            }
        }
        if (!hasPositive) {
            for (int i = 0; i < weights.length; i++) {
                weights[i] = 1.0 / weights.length;
            }
            return weights;
        }
        normalize(weights);
        return weights;
    }

    private static double[] solveLinearSystem(double[][] matrix, double[] rhs) {
        int n = rhs.length;
        double[][] a = new double[n][n + 1];
        for (int i = 0; i < n; i++) {
            System.arraycopy(matrix[i], 0, a[i], 0, n);
            a[i][n] = rhs[i];
        }
        for (int col = 0; col < n; col++) {
            int pivot = col;
            for (int row = col + 1; row < n; row++) {
                if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) {
                    pivot = row;
                }
            }
            double[] tmp = a[col];
            a[col] = a[pivot];
            a[pivot] = tmp;
            if (Math.abs(a[col][col]) < 1e-12) {
                a[col][col] = 1e-12;
            }
            double div = a[col][col];
            for (int j = col; j <= n; j++) {
                a[col][j] /= div;
            }
            for (int row = 0; row < n; row++) {
                if (row == col) {
                    continue;
                }
                double factor = a[row][col];
                for (int j = col; j <= n; j++) {
                    a[row][j] -= factor * a[col][j];
                }
            }
        }
        double[] solution = new double[n];
        for (int i = 0; i < n; i++) {
            solution[i] = a[i][n];
        }
        return solution;
    }

    private static double[][] copy(double[][] source) {
        double[][] result = new double[source.length][source[0].length];
        for (int i = 0; i < source.length; i++) {
            System.arraycopy(source[i], 0, result[i], 0, source[i].length);
        }
        return result;
    }

    private static void normalize(double[] values) {
        double sum = 0.0;
        for (double value : values) {
            sum += value;
        }
        if (sum == 0.0) {
            double uniform = 1.0 / values.length;
            for (int i = 0; i < values.length; i++) {
                values[i] = uniform;
            }
            return;
        }
        for (int i = 0; i < values.length; i++) {
            values[i] /= sum;
        }
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
