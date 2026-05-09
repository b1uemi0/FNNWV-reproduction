package ceka.FNNWV;

import ceka.common.NeighborUtils;
import ceka.converters.FileLoader;
import ceka.core.Dataset;
import weka.core.Utils;

public class RunNeighborSelfCheck {
    public static void main(String[] args) throws Exception {
        String datasetPath = args.length > 0 ? args[0] : "datasets/simulation/iris0.arff";
        Dataset dataset = FileLoader.loadFile(datasetPath);
        dataset = RunFnnwvMv.replaceMissingValues(dataset);
        RunFnnwvMv.simulateWorkerLabels(dataset, 5, "uniform");

        double[][] ranges = NeighborUtils.minMax(dataset);
        double[] distances = new double[dataset.getExampleSize()];
        for (int j = 0; j < dataset.getExampleSize(); j++) {
            distances[j] = NeighborUtils.heomDistance(dataset, ranges, 0, j);
        }
        int[] sorted = Utils.sort(distances);
        System.out.println("nearest_index_for_example_0: " + sorted[0]);
        System.out.println("nearest_is_self: " + (sorted[0] == 0));
        System.out.println("self_distance: " + distances[0]);
        System.out.println("second_nearest_index_for_example_0: " + sorted[1]);
        System.out.println("second_nearest_distance: " + distances[sorted[1]]);

        Dataset withSelf = RunFnnwvMv.copyDataset(dataset);
        new FNNWV().doInference(withSelf);
        Dataset withoutSelf = RunFnnwvMv.copyDataset(dataset);
        new FNNWVNoSelf().doInference(withoutSelf);
        System.out.println("FNNWV_with_self_F1: " + RunFnnwvMv.calF1score(withSelf));
        System.out.println("FNNWV_without_self_F1: " + RunFnnwvMv.calF1score(withoutSelf));
    }
}
