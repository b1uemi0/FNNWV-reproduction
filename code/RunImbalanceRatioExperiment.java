package ceka.FNNWV;

import java.io.File;
import java.io.FileWriter;
import java.text.DecimalFormat;
import java.util.Arrays;

import ceka.converters.FileLoader;
import ceka.core.Dataset;

public class RunImbalanceRatioExperiment {
    public static void main(String[] args) throws Exception {
        String dataDir = args.length > 0 ? args[0] : "datasets/simulation";
        String outputPath = args.length > 1 ? args[1] : "results/FNNWV_imbalance_ratio_experiment.tsv";
        int times = args.length > 2 ? Integer.parseInt(args[2]) : 10;
        File[] files = new File(dataDir).listFiles((dir, name) -> name.toLowerCase().endsWith(".arff"));
        Arrays.sort(files, (a, b) -> a.getName().compareToIgnoreCase(b.getName()));
        DecimalFormat df = new DecimalFormat("#.00");
        File outputFile = new File(outputPath);
        File parent = outputFile.getParentFile();
        if (parent != null && !parent.exists()) {
            parent.mkdirs();
        }
        try (FileWriter writer = new FileWriter(outputFile)) {
            writer.write("dataset\timbalance_ratio\tMV\tFNNWV\r\n");
            for (File file : files) {
                double mvSum = 0.0;
                double fnnwvSum = 0.0;
                double ratio = 0.0;
                for (int t = 0; t < times; t++) {
                    Dataset dataset = FileLoader.loadFile(file.getPath());
                    dataset = RunFnnwvMv.replaceMissingValues(dataset);
                    ratio = imbalanceRatio(dataset);
                    RunFnnwvMv.simulateWorkerLabels(dataset, 5, "uniform");
                    Dataset mvData = RunFnnwvMv.copyDataset(dataset);
                    new ceka.consensus.MajorityVote().doInference(mvData);
                    mvSum += RunFnnwvMv.calF1score(mvData);
                    Dataset fnnwvData = RunFnnwvMv.copyDataset(dataset);
                    new FNNWV().doInference(fnnwvData);
                    fnnwvSum += RunFnnwvMv.calF1score(fnnwvData);
                }
                writer.write(strip(file.getName()) + "\t" + df.format(ratio) + "\t"
                        + df.format(mvSum / times) + "\t" + df.format(fnnwvSum / times) + "\r\n");
                writer.flush();
            }
        }
        System.out.println("Wrote results to " + outputFile.getPath());
    }

    private static double imbalanceRatio(Dataset dataset) {
        int[] counts = new int[dataset.getCategorySize()];
        for (int i = 0; i < dataset.getExampleSize(); i++) {
            counts[dataset.getExampleByIndex(i).getTrueLabel().getValue()]++;
        }
        int min = Integer.MAX_VALUE;
        int max = 0;
        for (int count : counts) {
            if (count > 0 && count < min) {
                min = count;
            }
            if (count > max) {
                max = count;
            }
        }
        return (double) max / (double) min;
    }

    private static String strip(String name) {
        return name.replaceFirst("(?i)\\.arff$", "");
    }
}
