package ceka.FNNWV;

import java.io.File;
import java.io.FileWriter;
import java.text.DecimalFormat;

import ceka.converters.FileLoader;
import ceka.core.Dataset;

public class RunNerAblation {
    public static void main(String[] args) throws Exception {
        String dataDir = args.length > 0 ? args[0] : "datasets/real-world";
        String outputPath = args.length > 1 ? args[1] : "results/FNNWV_ner_ablation.tsv";
        File dir = new File(dataDir);
        Dataset dataset = FileLoader.loadFile(
                new File(dir, "NER.response.txt").getPath(),
                new File(dir, "NER.gold.txt").getPath(),
                new File(dir, "NER.arff").getPath());

        DecimalFormat df = new DecimalFormat("#.00");
        Dataset v0 = RunFnnwvMv.copyDataset(dataset);
        new FNNWVVariant(FNNWVVariant.Mode.MV_ONLY).doInference(v0);
        Dataset v1 = RunFnnwvMv.copyDataset(dataset);
        new FNNWVVariant(FNNWVVariant.Mode.NEAREST_ONLY).doInference(v1);
        Dataset v2 = RunFnnwvMv.copyDataset(dataset);
        new FNNWVVariant(FNNWVVariant.Mode.FARTHEST_ONLY).doInference(v2);
        Dataset full = RunFnnwvMv.copyDataset(dataset);
        new FNNWV().doInference(full);

        File outputFile = new File(outputPath);
        File parent = outputFile.getParentFile();
        if (parent != null && !parent.exists()) {
            parent.mkdirs();
        }
        try (FileWriter writer = new FileWriter(outputFile)) {
            writer.write("dataset\tFNNWV-0\tFNNWV-1\tFNNWV-2\tFNNWV\r\n");
            writer.write("NER\t" + df.format(RunFnnwvMv.calF1score(v0))
                    + "\t" + df.format(RunFnnwvMv.calF1score(v1))
                    + "\t" + df.format(RunFnnwvMv.calF1score(v2))
                    + "\t" + df.format(RunFnnwvMv.calF1score(full)) + "\r\n");
        }
        System.out.println("Wrote results to " + outputFile.getPath());
    }
}
