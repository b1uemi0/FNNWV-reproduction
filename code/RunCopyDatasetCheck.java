package ceka.FNNWV;

import ceka.converters.FileLoader;
import ceka.core.Dataset;

public class RunCopyDatasetCheck {
    public static void main(String[] args) throws Exception {
        String datasetPath = args.length > 0 ? args[0] : "datasets/simulation/iris0.arff";
        Dataset dataset = FileLoader.loadFile(datasetPath);
        RunFnnwvMv.simulateWorkerLabels(dataset, 5, "uniform");
        Dataset copy = RunFnnwvMv.copyDataset(dataset);

        System.out.println("Example: " + (dataset.getExampleByIndex(0) == copy.getExampleByIndex(0)));
        System.out.println("Worker: " + (dataset.getWorkerByIndex(0) == copy.getWorkerByIndex(0)));
        System.out.println("Category: " + (dataset.getCategory(0) == copy.getCategory(0)));
    }
}
