package ceka.FNNWV;

import java.io.File;
import java.io.FileWriter;
import java.text.DecimalFormat;

import ceka.consensus.MajorityVote;
import ceka.consensus.plat.PLAT;
import ceka.converters.FileLoader;
import ceka.core.Dataset;
import ceka.IWMV.IWMV;
import ceka.LAWMV.LAWMV;
import ceka.MNLDP.MNLDP;

public class RunRealWorldNer {
    public static void main(String[] args) throws Exception {
        String dataDir = args.length > 0 ? args[0] : "datasets/real-world";
        String outputPath = args.length > 1 ? args[1] : "results/FNNWV_realworld_ner_mv_fnnwv.tsv";

        File dir = new File(dataDir);
        File arff = new File(dir, "NER.arff");
        File response = new File(dir, "NER.response.txt");
        File gold = new File(dir, "NER.gold.txt");

        System.out.println("Loading NER data...");
        Dataset dataset = FileLoader.loadFile(response.getPath(), gold.getPath(), arff.getPath());
        System.out.println("Loaded examples=" + dataset.getExampleSize()
                + ", classes=" + dataset.getCategorySize()
                + ", workers=" + dataset.getWorkerSize()
                + ", attributes=" + dataset.numAttributes());
        DecimalFormat df = new DecimalFormat("#.00");

        System.out.println("Running MV...");
        Dataset mvData = RunFnnwvMv.copyDataset(dataset);
        MajorityVote mv = new MajorityVote();
        mv.doInference(mvData);
        System.out.println("MV minority F1: " + df.format(RunFnnwvMv.calF1score(mvData)));

        System.out.println("Running IWMV...");
        Dataset iwmvData = RunFnnwvMv.copyDataset(dataset);
        IWMV iwmv = new IWMV();
        iwmv.doInference(iwmvData);
        System.out.println("IWMV minority F1: " + df.format(RunFnnwvMv.calF1score(iwmvData)));

        System.out.println("Running PLAT...");
        Dataset platData = RunFnnwvMv.copyDataset(dataset);
        PLAT plat = new PLAT();
        plat.doInference(platData);
        System.out.println("PLAT minority F1: " + df.format(RunFnnwvMv.calF1score(platData)));

        System.out.println("Running MNLDP...");
        Dataset mnldpData = RunFnnwvMv.copyDataset(dataset);
        MNLDP mnldp = new MNLDP();
        mnldp.doInference(mnldpData);
        System.out.println("MNLDP minority F1: " + df.format(RunFnnwvMv.calF1score(mnldpData)));

        System.out.println("Running LAWMV...");
        Dataset lawmvData = RunFnnwvMv.copyDataset(dataset);
        LAWMV lawmv = new LAWMV();
        lawmv.doInference(lawmvData);
        System.out.println("LAWMV minority F1: " + df.format(RunFnnwvMv.calF1score(lawmvData)));

        System.out.println("Running FNNWV...");
        Dataset fnnwvData = RunFnnwvMv.copyDataset(dataset);
        FNNWVFast fnnwv = new FNNWVFast();
        fnnwv.doInference(fnnwvData);

        File outputFile = new File(outputPath);
        File parent = outputFile.getParentFile();
        if (parent != null && !parent.exists()) {
            parent.mkdirs();
        }
        try (FileWriter writer = new FileWriter(outputFile)) {
            writer.write("dataset\texamples\tclasses\tworkers\tMV\tIWMV\tPLAT\tMNLDP\tLAWMV\tFNNWV\r\n");
            writer.write("NER\t" + dataset.getExampleSize() + "\t" + dataset.getCategorySize() + "\t"
                    + dataset.getWorkerSize() + "\t" + df.format(RunFnnwvMv.calF1score(mvData))
                    + "\t" + df.format(RunFnnwvMv.calF1score(iwmvData))
                    + "\t" + df.format(RunFnnwvMv.calF1score(platData))
                    + "\t" + df.format(RunFnnwvMv.calF1score(mnldpData))
                    + "\t" + df.format(RunFnnwvMv.calF1score(lawmvData))
                    + "\t" + df.format(RunFnnwvMv.calF1score(fnnwvData)) + "\r\n");
        }

        System.out.println("dataset: NER");
        System.out.println("examples: " + dataset.getExampleSize());
        System.out.println("classes: " + dataset.getCategorySize());
        System.out.println("workers: " + dataset.getWorkerSize());
        System.out.println("MV minority F1: " + df.format(RunFnnwvMv.calF1score(mvData)));
        System.out.println("IWMV minority F1: " + df.format(RunFnnwvMv.calF1score(iwmvData)));
        System.out.println("PLAT minority F1: " + df.format(RunFnnwvMv.calF1score(platData)));
        System.out.println("MNLDP minority F1: " + df.format(RunFnnwvMv.calF1score(mnldpData)));
        System.out.println("LAWMV minority F1: " + df.format(RunFnnwvMv.calF1score(lawmvData)));
        System.out.println("FNNWV minority F1: " + df.format(RunFnnwvMv.calF1score(fnnwvData)));
        System.out.println("Wrote results to " + outputFile.getPath());
    }
}
