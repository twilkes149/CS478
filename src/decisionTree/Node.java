package decisionTree;

import java.util.HashSet;
import java.util.Iterator;

import labs.Matrix;

public class Node {
	
	private int splitIndex;//the feature index this node splits the data on
	private Node[] children;//a list of children, the index of the array corresponds to the value of the feature
	
	//inits the children array
	public void setChildrenSize(int size) {
		this.children = new Node[size];
	}
	
	/**
	 * 
	 * @param featureCol the column we are currently looking at
	 * @param value the value we are trying to find the probability of
	 * @param set the whole data set for this node
	 * @return probability the probability that value of featureCol is in set
	 */
	public double probability(int featureCol, double value, Matrix set) {
		double numerator = set.rows();
		double count = 0;
		
		for (int i = 0; i < set.rows(); i++) {//for all rows
			if (set.get(i, featureCol) == value) {//if the column in this row == the value passed in, inc count
				count++;
			}
		}
		return count/numerator;//return probability
	}
	/**
	 * This function implements the attribute info function
	 * @param features the data set
	 * @param colIndex the feature to score
	 * @param labels the output class for the feature, where each row corresponds to each row in the features 
	 * @return the info for this feature
	 * @throws Exception
	 */
	public double info(Matrix features, int colIndex, Matrix labels) throws Exception {		
		HashSet<Double> values = (HashSet<Double>) features.getFeatureValues(features, colIndex);//gets all values for this column
		
		double sum = 0;
		for (Iterator<Double> i = values.iterator(); i.hasNext();) {//for all the values
			double temp = i.next();
			double prob = this.probability(colIndex, temp, features);			
						
			int numValues = features.countVal(colIndex, temp);			
			
			Matrix sub[] = features.select(features, colIndex, temp, labels, numValues);//sub[0] will have the selected features, sub[1] will have the labels to match
			
			//System.out.print(numValues + "/" + features.rows() + "*(");
			double firstTerm = numValues/((double) features.rows());
			double tempSum=0;
			
			double[] valuesCount = sub[1].countUnique(sub[1], 0);
			
			for (int j = 0; j < valuesCount.length; j++) {
				//System.out.print(" - "+valuesCount[j]+"/"+numValues + "*log(" +valuesCount[j] +"/"+numValues+")");
				tempSum -= (valuesCount[j]/numValues) * (Math.log((valuesCount[j]/numValues))/Math.log(2));
			}
			
			sum += firstTerm*tempSum;
			//System.out.println(")");
			//System.out.println();
			
		}
		return sum;
	}
}
