package ceka.FNNWV;

import java.io.File;
import java.io.FileWriter;
import java.text.DecimalFormat;
import java.util.Arrays;

import ceka.converters.FileLoader;
import ceka.core.Dataset;

public class RunWorkersExperiment {
    public static void main(String[] args) throws Exception {
        String dataDir = args.length > 0 ? args[0] : "datasets/simulation";
        String outputPath = args.length > 1 ? args[1] : "results/FNNWV_workers_experiment.tsv";
        int times = args.length > 2 ? Integer.parseInt(args[2]) : 10;
        int[] workersList = args.length > 3 ? parseWorkers(args[3]) : new int[] {3, 5, 7, 9, 11};

        File[] files = new File(dataDir).listFiles((dir, name) -> name.toLowerCase().endsWith(".arff"));
        Arrays.sort(files, (a, b) -> a.getName().compareToIgnoreCase(b.getName()));
        DecimalFormat df = new DecimalFormat("#.00");
        File outputFile = new File(outputPath);
        File parent = outputFile.getParentFile();
        if (parent != null && !parent.exists()) {
            parent.mkdirs();
        }
        try (FileWriter writer = new FileWriter(outputFile)) {
            writer.write("workers\tMV\tFNNWV\r\n");
            for (int workers : workersList) {
                double mvSum = 0.0;
                double fnnwvSum = 0.0;
                for (File file : files) {
                    for (int t = 0; t < times; t++) {
                        Dataset dataset = FileLoader.loadFile(file.getPath());
                        dataset = RunFnnwvMv.replaceMissingValues(dataset);
                        RunFnnwvMv.simulateWorkerLabels(dataset, workers, "uniform");
                        Dataset mvData = RunFnnwvMv.copyDataset(dataset);
                        new ceka.consensus.MajorityVote().doInference(mvData);
                        mvSum += RunFnnwvMv.calF1score(mvData);
                        Dataset fnnwvData = RunFnnwvMv.copyDataset(dataset);
                        new FNNWV().doInference(fnnwvData);
                        fnnwvSum += RunFnnwvMv.calF1score(fnnwvData);
                    }
                }
                double denom = files.length * times;
                writer.write(workers + "\t" + df.format(mvSum / denom) + "\t" + df.format(fnnwvSum / denom) + "\r\n");
                writer.flush();
                System.out.println("workers=" + workers + " done");
            }
        }
        System.out.println("Wrote results to " + outputFile.getPath());
    }

    private static int[] parseWorkers(String value) {
        String[] parts = value.split(",");
        int[] result = new int[parts.length];
        for (int i = 0; i < parts.length; i++) {
            result[i] = Integer.parseInt(parts[i].trim());
        }
        return result;
    }
}
