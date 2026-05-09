// 最远最近邻加权MV
package ceka.FNNWV;

import ceka.core.Category;
import ceka.core.Dataset;
import ceka.core.Example;
import ceka.core.Label;
import ceka.core.MultiNoisyLabelSet;
import ceka.core.Worker;
import weka.core.Utils;

public class FNNWV{
	public static final String NAME = "FNNWV";
	public void doInference(Dataset dataset) throws Exception {
		// 实例的个数
		int numExample = dataset.getExampleSize();
		int m_KNN = (int)(numExample / dataset.getCategorySize() * 0.5);
		
		// 每个属性的最大值和最小值
		double attValueMax[] = new double[dataset.numAttributes()];
		double attValueMin[] = new double[dataset.numAttributes()];
		
		// 找每个属性的最大值和最小值
		for(int i = 0; i < dataset.numAttributes(); i++){
			if(i != dataset.classIndex()){
				if(dataset.attribute(i).isNumeric()){
					double[] attValue = new double[numExample];
					for (int j = 0; j < numExample; j++) {
						attValue[j] = dataset.getExampleByIndex(j).value(i);
					}
					attValueMin[i] = attValue[Utils.minIndex(attValue)];
					attValueMax[i] = attValue[Utils.maxIndex(attValue)];
				}
			}
		}
		
		// 为每一个实例找K个近邻和远邻
		// 计算距离，HEOM距离
		for(int i = 0; i < numExample; i++){
			int[] indexDistanceSort = new int[numExample];
			double[] distance = new double[numExample];
			double[] weight = new double[numExample];
			double[] weight_sum = new double[numExample];
			// 目标实例
			Example e = dataset.getExampleByIndex(i);
			MultiNoisyLabelSet mnls = dataset.getExampleByIndex(i).getMultipleNoisyLabelSet(0);
			// 计算其它实例到目标实例的距离，并分别利用工人偏置和实例的距离计算权重
			for(int j = 0; j < numExample; j++){
				Example e1 = dataset.getExampleByIndex(j);
				// 利用工人偏置计算权重
				for(int k = 0; k < mnls.getLabelSetSize(); k++){
					Label label = e1.getNoisyLabelByWorkerId(mnls.getLabel(k).getWorkerId());
					if(label != null){
						weight_sum[j]++;
						if(label.getValue() == mnls.getLabel(k).getValue()){
							weight[j] += 1.0;
						}
					}
				}
				// 考虑当分母不为0的情况
				if(weight_sum[j] == 0){
					weight[j] = 0;
				}else{
					weight[j] = weight[j] / weight_sum[j];
				}				
				// 计算距离
				for (int k = 0; k < dataset.numAttributes(); k++) {
					if (k != dataset.classIndex()){
						// 属性值缺失,为1
						if (e.isMissing(k)||e1.isMissing(k)){
							distance[j] += 1;
						}
						else if (dataset.attribute(k).isNominal()){ //名词属性：OM距离
							if (e.value(k) != e1.value(k)) {
								distance[j] += 1.0;
							}
						}
						else if (dataset.attribute(k).isNumeric()){ //数值属性：欧氏距离
							double max_min = (attValueMax[k] - attValueMin[k]);
							if(max_min != 0.0){
								distance[j] += Math.pow((Math.abs(e.value(k)-e1.value(k)) / max_min),2);
							}
						}
					}
				}		
				//距离开方
				distance[j] = Math.sqrt(distance[j]);
			}
			//根据距离排序
			indexDistanceSort = Utils.sort(distance);
			
			int numClasses = dataset.numClasses();
			double[] classCounts = new double[numClasses];
			
			// 先累加近邻
			for (int j = 0; j < m_KNN; j++) {
				MultiNoisyLabelSet mnlsn = dataset.getExampleByIndex(indexDistanceSort[j]).getMultipleNoisyLabelSet(0);
				double tempWight = weight[indexDistanceSort[j]] + (1.0 - (distance[indexDistanceSort[j]] / distance[indexDistanceSort[m_KNN-1]]));
				tempWight *= 0.5;
				for(int k = 0; k < mnlsn.getLabelSetSize(); k++){
					classCounts[mnlsn.getLabel(k).getValue()] += tempWight;
				}	
			}
			// 计算两类差值
			double temp = Math.abs(classCounts[0] - classCounts[1]) / Utils.sum(classCounts);
			if(temp < 0.1) {
				// 再删去远的
				for (int j = 0; j < m_KNN; j++) {
					int temp_index = indexDistanceSort.length - 1 - j;
					MultiNoisyLabelSet mnlsn = dataset.getExampleByIndex(indexDistanceSort[temp_index]).getMultipleNoisyLabelSet(0);
					double tempWight = (1-weight[indexDistanceSort[temp_index]]) + ((distance[indexDistanceSort[temp_index]] / distance[indexDistanceSort[indexDistanceSort.length - 1]]));
					tempWight *= 0.5;
					for(int k = 0; k < mnlsn.getLabelSetSize(); k++){
						classCounts[mnlsn.getLabel(k).getValue()] -= tempWight;
					}	
				}
			}
		
			int maxIndex = 0;
			double maxValue = -10000.0;
			for (int j = 0; j < numClasses; j++) {
				if(classCounts[j] > maxValue){
					maxValue = classCounts[j];
					maxIndex = j;
				}
			}
	
			Label label = new Label(null, String.valueOf(maxIndex), e.getId(), NAME);
			e.setIntegratedLabel(label);
		}
		dataset.assignIntegeratedLabel2WekaInstanceClassValue();
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

}
