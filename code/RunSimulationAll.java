package ceka.FNNWV;

import java.io.File;
import java.io.FileWriter;
import java.text.DecimalFormat;
import java.util.Arrays;

import ceka.consensus.MajorityVote;
import ceka.consensus.plat.PLAT;
import ceka.converters.FileLoader;
import ceka.core.Dataset;
import ceka.IWMV.IWMV;
import ceka.LAWMV.LAWMV;
import ceka.MNLDP.MNLDP;

public class RunSimulationAll {
    private static final int DEFAULT_TIMES = 10;
    private static final int DEFAULT_WORKERS = 5;

    public static void main(String[] args) throws Exception {
        String dataDir = args.length > 0 ? args[0] : "datasets/simulation";
        String outputPath = args.length > 1 ? args[1] : "results/FNNWV_simulation_mv_fnnwv.tsv";
        int times = args.length > 2 ? Integer.parseInt(args[2]) : DEFAULT_TIMES;
        int workers = args.length > 3 ? Integer.parseInt(args[3]) : DEFAULT_WORKERS;
        String distribution = args.length > 4 ? args[4] : "uniform";

        File[] files = new File(dataDir).listFiles((dir, name) -> name.toLowerCase().endsWith(".arff"));
        if (files == null || files.length == 0) {
            throw new IllegalArgumentException("No ARFF files found under " + dataDir);
        }
        Arrays.sort(files, (a, b) -> a.getName().compareToIgnoreCase(b.getName()));

        File outputFile = new File(outputPath);
        File parent = outputFile.getParentFile();
        if (parent != null && !parent.exists()) {
            parent.mkdirs();
        }

        DecimalFormat df = new DecimalFormat("#.00");
        double allMv = 0.0;
        double allIwmv = 0.0;
        double allPlat = 0.0;
        double allMnldp = 0.0;
        double allLawmv = 0.0;
        double allFnnwv = 0.0;

        try (FileWriter writer = new FileWriter(outputFile)) {
            writer.write("dataset\texamples\tclasses\tMV\tIWMV\tPLAT\tMNLDP\tLAWMV\tFNNWV\r\n");
            for (File file : files) {
                System.out.println("Running " + file.getName());
                double resultMv = 0.0;
                double resultIwmv = 0.0;
                double resultPlat = 0.0;
                double resultMnldp = 0.0;
                double resultLawmv = 0.0;
                double resultFnnwv = 0.0;
                int examples = 0;
                int classes = 0;

                for (int round = 0; round < times; round++) {
                    Dataset dataset = FileLoader.loadFile(file.getPath());
                    dataset = RunFnnwvMv.replaceMissingValues(dataset);
                    RunFnnwvMv.simulateWorkerLabels(dataset, workers, distribution);
                    examples = dataset.getExampleSize();
                    classes = dataset.getCategorySize();

                    Dataset mvData = RunFnnwvMv.copyDataset(dataset);
                    MajorityVote mv = new MajorityVote();
                    mv.doInference(mvData);
                    resultMv += RunFnnwvMv.calF1score(mvData);

                    Dataset iwmvData = RunFnnwvMv.copyDataset(dataset);
                    IWMV iwmv = new IWMV();
                    iwmv.doInference(iwmvData);
                    resultIwmv += RunFnnwvMv.calF1score(iwmvData);

                    Dataset platData = RunFnnwvMv.copyDataset(dataset);
                    PLAT plat = new PLAT();
                    plat.doInference(platData);
                    resultPlat += RunFnnwvMv.calF1score(platData);

                    Dataset mnldpData = RunFnnwvMv.copyDataset(dataset);
                    MNLDP mnldp = new MNLDP();
                    mnldp.doInference(mnldpData);
                    resultMnldp += RunFnnwvMv.calF1score(mnldpData);

                    Dataset lawmvData = RunFnnwvMv.copyDataset(dataset);
                    LAWMV lawmv = new LAWMV();
                    lawmv.doInference(lawmvData);
                    resultLawmv += RunFnnwvMv.calF1score(lawmvData);

                    Dataset fnnwvData = RunFnnwvMv.copyDataset(dataset);
                    FNNWV fnnwv = new FNNWV();
                    fnnwv.doInference(fnnwvData);
                    resultFnnwv += RunFnnwvMv.calF1score(fnnwvData);
                }

                double avgMv = resultMv / times;
                double avgIwmv = resultIwmv / times;
                double avgPlat = resultPlat / times;
                double avgMnldp = resultMnldp / times;
                double avgLawmv = resultLawmv / times;
                double avgFnnwv = resultFnnwv / times;
                allMv += avgMv;
                allIwmv += avgIwmv;
                allPlat += avgPlat;
                allMnldp += avgMnldp;
                allLawmv += avgLawmv;
                allFnnwv += avgFnnwv;
                writer.write(stripArff(file.getName()) + "\t" + examples + "\t" + classes + "\t"
                        + df.format(avgMv) + "\t" + df.format(avgIwmv) + "\t"
                        + df.format(avgPlat) + "\t" + df.format(avgMnldp) + "\t"
                        + df.format(avgLawmv) + "\t" + df.format(avgFnnwv) + "\r\n");
                writer.flush();
                System.out.println(stripArff(file.getName()) + "\tMV=" + df.format(avgMv)
                        + "\tIWMV=" + df.format(avgIwmv)
                        + "\tPLAT=" + df.format(avgPlat)
                        + "\tMNLDP=" + df.format(avgMnldp)
                        + "\tLAWMV=" + df.format(avgLawmv)
                        + "\tFNNWV=" + df.format(avgFnnwv));
            }

            writer.write("average\t\t\t" + df.format(allMv / files.length) + "\t"
                    + df.format(allIwmv / files.length) + "\t"
                    + df.format(allPlat / files.length) + "\t"
                    + df.format(allMnldp / files.length) + "\t"
                    + df.format(allLawmv / files.length) + "\t"
                    + df.format(allFnnwv / files.length) + "\r\n");
        }

        System.out.println("Wrote results to " + outputFile.getPath());
        System.out.println("worker_quality_distribution: " + distribution);
    }

    private static String stripArff(String name) {
        return name.replaceFirst("(?i)\\.arff$", "");
    }
}
