package ceka.FNNWV;

import java.util.ArrayList;
import java.util.Collections;
import java.io.File;
import java.text.DecimalFormat;
import java.util.Locale;
import java.util.Random;

import ceka.consensus.MajorityVote;
import ceka.converters.FileLoader;
import ceka.core.Category;
import ceka.core.Dataset;
import ceka.core.Example;
import ceka.core.Label;
import ceka.core.Worker;
import ceka.simulation.GaussianLabelingStrategy;
import ceka.simulation.MockWorker;
import ceka.simulation.SingleQualLabelingStrategy;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

public class RunFnnwvMv {
    private static final int DEFAULT_WORKERS = 5;

    public static void main(String[] args) throws Exception {
        String datasetPath = args.length > 0 ? args[0] : "datasets/simulation/iris0.arff";
        int workers = args.length > 1 ? Integer.parseInt(args[1]) : DEFAULT_WORKERS;
        String distribution = args.length > 2 ? args[2] : "uniform";

        Dataset dataset = FileLoader.loadFile(datasetPath);
        dataset = replaceMissingValues(dataset);
        simulateWorkerLabels(dataset, workers, distribution);

        Dataset mvData = copyDataset(dataset);
        MajorityVote mv = new MajorityVote();
        mv.doInference(mvData);

        Dataset fnnwvData = copyDataset(dataset);
        FNNWV fnnwv = new FNNWV();
        fnnwv.doInference(fnnwvData);

        DecimalFormat df = new DecimalFormat("#.00");
        System.out.println("dataset: " + new File(datasetPath).getPath());
        System.out.println("examples: " + dataset.getExampleSize());
        System.out.println("classes: " + dataset.getCategorySize());
        System.out.println("workers: " + workers);
        System.out.println("distribution: " + distribution);
        System.out.println("MV minority F1: " + df.format(calF1score(mvData)));
        System.out.println("FNNWV minority F1: " + df.format(calF1score(fnnwvData)));
    }

    public static void simulateWorkerLabels(Dataset dataset, int workers) {
        simulateWorkerLabels(dataset, workers, "normal");
    }

    public static void simulateWorkerLabels(Dataset dataset, int workers, String distribution) {
        String normalized = distribution.toLowerCase(Locale.ROOT);
        if ("uniform".equals(normalized)) {
            simulateUniformWorkerLabels(dataset, workers);
        } else if ("normal".equals(normalized) || "gaussian".equals(normalized)) {
            simulateGaussianWorkerLabels(dataset, workers);
        } else {
            throw new IllegalArgumentException("Unknown worker quality distribution: " + distribution);
        }
    }

    public static void simulateWorkerLabels(Dataset dataset, int workers, String distribution, long seed) {
        String normalized = distribution.toLowerCase(Locale.ROOT);
        Random random = new Random(seed);
        if ("uniform".equals(normalized)) {
            simulateSeededWorkerLabels(dataset, workers, random, false);
        } else if ("normal".equals(normalized) || "gaussian".equals(normalized)) {
            simulateSeededWorkerLabels(dataset, workers, random, true);
        } else {
            throw new IllegalArgumentException("Unknown worker quality distribution: " + distribution);
        }
    }

    private static void simulateGaussianWorkerLabels(Dataset dataset, int workers) {
        MockWorker[] mockWorkers = new MockWorker[workers];
        GaussianLabelingStrategy strategy = new GaussianLabelingStrategy(0.65, 0.15);
        for (int i = 0; i < mockWorkers.length; i++) {
            mockWorkers[i] = new MockWorker(String.valueOf(i + mockWorkers.length));
        }
        strategy.assignWorkerQuality(mockWorkers);
        for (MockWorker mockWorker : mockWorkers) {
            mockWorker.labeling(dataset, strategy);
        }
    }

    private static void simulateUniformWorkerLabels(Dataset dataset, int workers) {
        double max = 0.8000;
        double min = 0.5000;
        for (int i = 0; i < workers; i++) {
            double quality = Math.random() * (max - min) + min;
            SingleQualLabelingStrategy strategy = new SingleQualLabelingStrategy(quality);
            MockWorker mockWorker = new MockWorker(String.valueOf(i));
            mockWorker.setSingleQuality(quality);
            mockWorker.labeling(dataset, strategy);
        }
    }

    private static void simulateSeededWorkerLabels(Dataset dataset, int workers, Random random, boolean gaussian) {
        for (int i = 0; i < workers; i++) {
            double quality;
            if (gaussian) {
                quality = Math.max(0.0, Math.min(1.0, random.nextGaussian() * 0.15 + 0.65));
            } else {
                quality = random.nextDouble() * (0.8000 - 0.5000) + 0.5000;
            }
            Worker worker = new Worker(String.valueOf(i));
            dataset.addWorker(worker);
            labelWithSeededSingleQuality(dataset, worker, quality, random);
        }
    }

    private static void labelWithSeededSingleQuality(Dataset dataset, Worker worker, double quality, Random random) {
        int numCategory = dataset.getCategorySize();
        ArrayList<ArrayList<Example>> examplesByCategory = new ArrayList<ArrayList<Example>>();
        for (int i = 0; i < numCategory; i++) {
            examplesByCategory.add(new ArrayList<Example>());
        }
        for (int i = 0; i < dataset.getExampleSize(); i++) {
            Example example = dataset.getExampleByIndex(i);
            examplesByCategory.get(example.getTrueLabel().getValue()).add(example);
        }

        for (int category = 0; category < numCategory; category++) {
            ArrayList<Example> shuffled = new ArrayList<Example>(examplesByCategory.get(category));
            Collections.shuffle(shuffled, random);
            int correct = (int) (quality * shuffled.size());
            ArrayList<Example> errorExamples = new ArrayList<Example>();
            for (int i = 0; i < shuffled.size(); i++) {
                Example example = shuffled.get(i);
                if (i < correct) {
                    addNoisyLabel(worker, example, category);
                } else {
                    errorExamples.add(example);
                }
            }

            ArrayList<Integer> errorCategories = buildErrorCategories(examplesByCategory, category, errorExamples.size());
            Collections.shuffle(errorCategories, random);
            for (int i = 0; i < errorExamples.size(); i++) {
                addNoisyLabel(worker, errorExamples.get(i), errorCategories.get(i).intValue());
            }
        }
    }

    private static ArrayList<Integer> buildErrorCategories(ArrayList<ArrayList<Example>> examplesByCategory,
            int trueCategory, int errorSize) {
        ArrayList<Integer> errorCategories = new ArrayList<Integer>();
        int numCategory = examplesByCategory.size();
        int remainSize = 0;
        for (int i = 0; i < numCategory; i++) {
            if (i != trueCategory) {
                remainSize += examplesByCategory.get(i).size();
            }
        }
        int assigned = 0;
        for (int i = 0; i < numCategory; i++) {
            if (i == trueCategory) {
                continue;
            }
            int count;
            if (assigned == errorSize) {
                count = 0;
            } else if (isLastOtherCategory(i, trueCategory, numCategory)) {
                count = errorSize - assigned;
            } else {
                count = remainSize == 0 ? 0
                        : (int) (errorSize * ((double) examplesByCategory.get(i).size() / (double) remainSize));
                assigned += count;
            }
            for (int j = 0; j < count; j++) {
                errorCategories.add(new Integer(i));
            }
        }
        while (errorCategories.size() < errorSize) {
            errorCategories.add(new Integer((trueCategory + 1) % numCategory));
        }
        while (errorCategories.size() > errorSize) {
            errorCategories.remove(errorCategories.size() - 1);
        }
        return errorCategories;
    }

    private static boolean isLastOtherCategory(int index, int trueCategory, int numCategory) {
        for (int i = index + 1; i < numCategory; i++) {
            if (i != trueCategory) {
                return false;
            }
        }
        return true;
    }

    private static void addNoisyLabel(Worker worker, Example example, int value) {
        Label noisyLabel = new Label(null, String.valueOf(value), example.getId(), worker.getId());
        worker.addNoisyLabel(noisyLabel);
        example.addNoisyLabel(noisyLabel);
    }

    public static Dataset replaceMissingValues(Dataset dataset) throws Exception {
        ReplaceMissingValues missing = new ReplaceMissingValues();
        missing.setInputFormat(dataset);
        Instances instances = Filter.useFilter(dataset, missing);
        return instancesToDataset(instances, dataset);
    }

    public static Dataset copyDataset(Dataset dataset) {
        Dataset copyDataset = new Dataset(dataset, 0);
        for (int k = 0; k < dataset.getExampleSize(); k++) {
            Example example = dataset.getExampleByIndex(k);
            copyDataset.addExample(example);
        }
        for (int k = 0; k < dataset.getCategorySize(); k++) {
            Category category = dataset.getCategory(k);
            copyDataset.addCategory(category);
        }
        for (int k = 0; k < dataset.getWorkerSize(); k++) {
            Worker worker = dataset.getWorkerByIndex(k);
            copyDataset.addWorker(worker);
        }
        return copyDataset;
    }

    public static Dataset instancesToDataset(Instances instances, Dataset sourceDataset) {
        Dataset dataset = new Dataset(instances, instances.numInstances());
        for (int m = 0; m < sourceDataset.getCategorySize(); m++) {
            Category category = sourceDataset.getCategory(m);
            dataset.addCategory(category.copy());
        }
        for (int i = 0; i < instances.numInstances(); i++) {
            Instance instance = instances.instance(i);
            Integer trueValue = (int) instance.classValue();
            Example example = new Example(instance);
            Label trueLabel = new Label(null, trueValue.toString(), example.getId(), "create");
            example.setTrueLabel(trueLabel);
            dataset.addExample(example);
        }
        return dataset;
    }

    public static double calF1score(Dataset dataset) {
        int numExample = dataset.getExampleSize();
        int numClasses = dataset.getCategorySize();
        int minIndex = 0;
        int maxIndex = 0;
        Dataset[] perClass = new Dataset[numClasses];
        for (int j = 0; j < numClasses; j++) {
            perClass[j] = new Dataset(dataset, 0);
        }
        for (int m = 0; m < numExample; m++) {
            Example example = dataset.getExampleByIndex(m);
            perClass[example.getTrueLabel().getValue()].addExample(example);
        }
        for (int j = 0; j < numClasses; j++) {
            if (perClass[j].getExampleSize() > perClass[maxIndex].getExampleSize()) {
                maxIndex = j;
            }
            if (perClass[j].getExampleSize() < perClass[minIndex].getExampleSize()
                    && perClass[j].getExampleSize() != 0) {
                minIndex = j;
            }
        }

        double predictedMinority = 0.0;
        double truePositive = 0.0;
        double recalledMinority = 0.0;
        for (int i = 0; i < numExample; i++) {
            Example example = dataset.getExampleByIndex(i);
            if (example.getIntegratedLabel().getValue() == minIndex) {
                predictedMinority += 1.0;
                if (example.getTrueLabel().getValue() == minIndex) {
                    truePositive += 1.0;
                }
            }
            if (example.getTrueLabel().getValue() == minIndex
                    && example.getIntegratedLabel().getValue() == minIndex) {
                recalledMinority += 1.0;
            }
        }

        if (predictedMinority == 0.0 || perClass[minIndex].getExampleSize() == 0 || truePositive == 0.0) {
            return 0.0;
        }
        double precision = truePositive / predictedMinority * 100.0;
        double recall = recalledMinority / perClass[minIndex].getExampleSize() * 100.0;
        return 2.0 * precision * recall / (precision + recall);
    }
}
