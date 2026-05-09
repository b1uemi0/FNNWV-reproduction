package ceka.FNNWV;

import java.io.File;
import java.io.FileWriter;
import java.text.DecimalFormat;
import java.util.Arrays;

import ceka.LAWMV.LAWMV;
import ceka.consensus.MajorityVote;
import ceka.converters.FileLoader;
import ceka.core.Dataset;

public class RunAlphaExperiment {
    private static final double[] DEFAULT_ALPHAS = new double[] {0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30};
    private static final double[] DEFAULT_QUANTILES = new double[] {0.10, 0.20, 0.30};
    private static final long DEFAULT_BASE_SEED = 20260508L;

    public static void main(String[] args) throws Exception {
        String dataDir = args.length > 0 ? args[0] : "datasets/simulation";
        String runsPath = args.length > 1 ? args[1] : "results/FNNWV_alpha_experiment_runs.tsv";
        String summaryPath = args.length > 2 ? args[2] : defaultSummaryPath(runsPath);
        int times = args.length > 3 ? Integer.parseInt(args[3]) : 10;
        int workers = args.length > 4 ? Integer.parseInt(args[4]) : 5;
        String distribution = args.length > 5 ? args[5] : "uniform";
        long baseSeed = args.length > 6 ? Long.parseLong(args[6]) : DEFAULT_BASE_SEED;
        double[] alphas = args.length > 7 ? parseDoubles(args[7]) : DEFAULT_ALPHAS;
        double[] quantiles = args.length > 8 ? parseDoubles(args[8]) : DEFAULT_QUANTILES;

        File[] files = new File(dataDir).listFiles((dir, name) -> name.toLowerCase().endsWith(".arff"));
        if (files == null || files.length == 0) {
            throw new IllegalArgumentException("No ARFF files found under " + dataDir);
        }
        Arrays.sort(files, (a, b) -> a.getName().compareToIgnoreCase(b.getName()));

        ensureParent(runsPath);
        ensureParent(summaryPath);

        MethodSpec[] methods = buildMethods(alphas, quantiles);
        DecimalFormat df = new DecimalFormat("0.0000");
        double[][] datasetMeans = new double[methods.length][files.length];
        double[][] datasetTriggerMeans = new double[methods.length][files.length];
        double[][] datasetThresholdMeans = new double[methods.length][files.length];

        try (FileWriter runsWriter = new FileWriter(new File(runsPath));
                FileWriter summaryWriter = new FileWriter(new File(summaryPath))) {
            runsWriter.write("dataset\trun\tseed\tmode\tparameter\texamples\tclasses\tF1\ttrigger_rate\tthreshold\r\n");
            summaryWriter.write("dataset\tmode\tparameter\texamples\tclasses\tmean_F1\tstd_F1\tmean_trigger_rate\tstd_trigger_rate\tmean_threshold\tstd_threshold\r\n");

            for (int fileIndex = 0; fileIndex < files.length; fileIndex++) {
                File file = files[fileIndex];
                String datasetName = stripArff(file.getName());
                System.out.println("Running alpha experiment on " + file.getName());

                double[][] f1 = new double[methods.length][times];
                double[][] triggerRate = new double[methods.length][times];
                double[][] threshold = new double[methods.length][times];
                boolean[] hasStats = new boolean[methods.length];
                int examples = 0;
                int classes = 0;

                for (int run = 0; run < times; run++) {
                    long seed = baseSeed + fileIndex * 1000L + run;
                    Dataset dataset = loadSimulatedDataset(file, workers, distribution, seed);
                    examples = dataset.getExampleSize();
                    classes = dataset.getCategorySize();

                    for (int methodIndex = 0; methodIndex < methods.length; methodIndex++) {
                        Eval eval = evaluate(dataset, methods[methodIndex]);
                        f1[methodIndex][run] = eval.f1;
                        triggerRate[methodIndex][run] = eval.triggerRate;
                        threshold[methodIndex][run] = eval.threshold;
                        hasStats[methodIndex] = hasStats[methodIndex] || !Double.isNaN(eval.triggerRate);

                        runsWriter.write(datasetName + "\t" + run + "\t" + seed + "\t"
                                + methods[methodIndex].mode + "\t" + df.format(methods[methodIndex].parameter)
                                + "\t" + examples + "\t" + classes
                                + "\t" + df.format(eval.f1)
                                + "\t" + (Double.isNaN(eval.triggerRate) ? "" : df.format(eval.triggerRate))
                                + "\t" + (Double.isNaN(eval.threshold) ? "" : df.format(eval.threshold))
                                + "\r\n");
                    }
                }

                for (int methodIndex = 0; methodIndex < methods.length; methodIndex++) {
                    Summary f1Summary = summarize(f1[methodIndex]);
                    Summary triggerSummary = hasStats[methodIndex] ? summarize(triggerRate[methodIndex]) : Summary.empty();
                    Summary thresholdSummary = hasStats[methodIndex] ? summarize(threshold[methodIndex]) : Summary.empty();
                    datasetMeans[methodIndex][fileIndex] = f1Summary.mean;
                    datasetTriggerMeans[methodIndex][fileIndex] = triggerSummary.mean;
                    datasetThresholdMeans[methodIndex][fileIndex] = thresholdSummary.mean;
                    writeSummary(summaryWriter, datasetName, methods[methodIndex], examples, classes,
                            f1Summary, triggerSummary, thresholdSummary, df);
                }
                runsWriter.flush();
                summaryWriter.flush();
            }

            for (int methodIndex = 0; methodIndex < methods.length; methodIndex++) {
                Summary f1Summary = summarize(datasetMeans[methodIndex]);
                Summary triggerSummary = methods[methodIndex].hasTriggerStats()
                        ? summarize(datasetTriggerMeans[methodIndex]) : Summary.empty();
                Summary thresholdSummary = methods[methodIndex].hasTriggerStats()
                        ? summarize(datasetThresholdMeans[methodIndex]) : Summary.empty();
                writeSummary(summaryWriter, "average", methods[methodIndex], 0, 0,
                        f1Summary, triggerSummary, thresholdSummary, df);
            }
        }

        System.out.println("Wrote run-level results to " + runsPath);
        System.out.println("Wrote summary results to " + summaryPath);
        System.out.println("base_seed: " + baseSeed);
    }

    private static MethodSpec[] buildMethods(double[] alphas, double[] quantiles) {
        MethodSpec[] methods = new MethodSpec[3 + alphas.length + quantiles.length];
        methods[0] = new MethodSpec("MV", 0.0);
        methods[1] = new MethodSpec("LAWMV", 0.0);
        methods[2] = new MethodSpec("original", 0.1);
        int cursor = 3;
        for (double alpha : alphas) {
            methods[cursor++] = new MethodSpec("fixed_alpha", alpha);
        }
        for (double quantile : quantiles) {
            methods[cursor++] = new MethodSpec("adaptive_quantile", quantile);
        }
        return methods;
    }

    private static Eval evaluate(Dataset dataset, MethodSpec method) throws Exception {
        Dataset data = RunFnnwvMv.copyDataset(dataset);
        Eval eval = new Eval();
        if ("MV".equals(method.mode)) {
            new MajorityVote().doInference(data);
            eval.f1 = RunFnnwvMv.calF1score(data);
            eval.triggerRate = Double.NaN;
            eval.threshold = Double.NaN;
            return eval;
        }
        if ("LAWMV".equals(method.mode)) {
            new LAWMV().doInference(data);
            eval.f1 = RunFnnwvMv.calF1score(data);
            eval.triggerRate = Double.NaN;
            eval.threshold = Double.NaN;
            return eval;
        }
        if ("original".equals(method.mode)) {
            new FNNWV().doInference(data);
            eval.f1 = RunFnnwvMv.calF1score(data);
            eval.triggerRate = Double.NaN;
            eval.threshold = Double.NaN;
            return eval;
        }

        FNNWVAlpha fnnwv = "adaptive_quantile".equals(method.mode)
                ? new FNNWVAlpha(FNNWVAlpha.ThresholdMode.QUANTILE, method.parameter, "FNNWV-Q" + method.parameter)
                : new FNNWVAlpha(method.parameter);
        fnnwv.doInference(data);
        FNNWVAlpha.Stats stats = fnnwv.getLastStats();
        eval.f1 = RunFnnwvMv.calF1score(data);
        eval.triggerRate = stats.triggerRate();
        eval.threshold = stats.threshold;
        return eval;
    }

    private static Dataset loadSimulatedDataset(File file, int workers, String distribution, long seed) throws Exception {
        Dataset dataset = FileLoader.loadFile(file.getPath());
        dataset = RunFnnwvMv.replaceMissingValues(dataset);
        RunFnnwvMv.simulateWorkerLabels(dataset, workers, distribution, seed);
        return dataset;
    }

    private static void writeSummary(FileWriter writer, String dataset, MethodSpec method, int examples, int classes,
            Summary f1, Summary triggerRate, Summary threshold, DecimalFormat df) throws Exception {
        writer.write(dataset + "\t" + method.mode + "\t" + df.format(method.parameter)
                + "\t" + examples + "\t" + classes
                + "\t" + df.format(f1.mean)
                + "\t" + df.format(f1.std)
                + "\t" + (triggerRate.present ? df.format(triggerRate.mean) : "")
                + "\t" + (triggerRate.present ? df.format(triggerRate.std) : "")
                + "\t" + (threshold.present ? df.format(threshold.mean) : "")
                + "\t" + (threshold.present ? df.format(threshold.std) : "")
                + "\r\n");
    }

    private static Summary summarize(double[] values) {
        double mean = 0.0;
        for (double value : values) {
            mean += value;
        }
        mean = values.length == 0 ? 0.0 : mean / values.length;
        double variance = 0.0;
        for (double value : values) {
            variance += Math.pow(value - mean, 2);
        }
        double std = values.length <= 1 ? 0.0 : Math.sqrt(variance / (values.length - 1));
        Summary summary = new Summary();
        summary.mean = mean;
        summary.std = std;
        summary.present = true;
        return summary;
    }

    private static double[] parseDoubles(String value) {
        String[] parts = value.split(",");
        double[] result = new double[parts.length];
        for (int i = 0; i < parts.length; i++) {
            result[i] = Double.parseDouble(parts[i].trim());
        }
        return result;
    }

    private static void ensureParent(String path) {
        File file = new File(path);
        File parent = file.getParentFile();
        if (parent != null && !parent.exists()) {
            parent.mkdirs();
        }
    }

    private static String defaultSummaryPath(String runsPath) {
        return runsPath.replaceFirst("(?i)\\.tsv$", "_summary.tsv");
    }

    private static String stripArff(String name) {
        return name.replaceFirst("(?i)\\.arff$", "");
    }

    private static class MethodSpec {
        String mode;
        double parameter;

        MethodSpec(String mode, double parameter) {
            this.mode = mode;
            this.parameter = parameter;
        }

        boolean hasTriggerStats() {
            return "fixed_alpha".equals(mode) || "adaptive_quantile".equals(mode);
        }
    }

    private static class Eval {
        double f1;
        double triggerRate;
        double threshold;
    }

    private static class Summary {
        double mean;
        double std;
        boolean present;

        static Summary empty() {
            return new Summary();
        }
    }
}
