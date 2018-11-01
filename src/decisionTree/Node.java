package decisionTree;

import java.util.HashSet;
import java.util.Iterator;

import labs.Matrix;

public class Node {
	
	private int splitIndex;//the feature index this node splits the data on
	private Node[] children;//a list of children, the index of the array corresponds to the value of the feature
	Matrix[] dataSet;//this is used for prediction
	private boolean leafNode;
	
	public static final boolean DEBUG = false;
	
	public Node() {
		dataSet = new Matrix[2];//this will hold one matrix for features, and one for labels
		leafNode = false;
		splitIndex = Integer.MAX_VALUE;//THIS USED TO BE MAX INT
		children = null;
	}
	
	public void setLeaf() {
		this.leafNode = true;
	}
	
	public boolean isLeafNode() {
		if (this.leafNode) {
			return true;  
		}
		else if (this.children == null){
			return true;
		}
		else return false;
	}
	
	//inits the children array
	public void setChildrenSize(int size) {
		this.children = new Node[size];
		for (int i = 0; i < size; i++) {
			this.children[i] = new Node();
			this.children[i].setDataSet(this.dataSet);//give the next node this node's data
			//this.children[i].splitIndex = this.splitIndex;//not sure if this will work.....
		}
	}
	
	public Matrix getData() {
		return this.dataSet[0];
	}
	
	public Matrix getLabels() {
		return this.dataSet[1];
	}
	
	public Node getChild(int index) {
		return children[index];
	}
	
	public void setDataSet(Matrix[] set) {
		this.dataSet = set;
	}
	
	public int getSplitIndex() {
		return splitIndex;
	}

	public void setSplitIndex(int splitIndex) {
		this.splitIndex = splitIndex;
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
		
		if (Node.DEBUG)
			System.out.println("calculating info for: " + features.getAttributeName(colIndex));
		
		double sum = 0;
		for (Iterator<Double> i = values.iterator(); i.hasNext();) {//for all the values
			double temp = i.next();
			//double prob = this.probability(colIndex, temp, features);			
						
			int numValues = features.countVal(colIndex, temp);			
			
			Matrix sub[] = features.select(features, colIndex, temp, labels, numValues);//sub[0] will have the selected features, sub[1] will have the labels to match
						
			if (Node.DEBUG)
				System.out.print(numValues + "/" + features.rows() + "*(");
			
			double firstTerm = numValues/((double) features.rows());
			double tempSum=0;
			
			double[] valuesCount = sub[1].countUnique(sub[1], 0);
			
			for (int j = 0; j < valuesCount.length; j++) {
				if (Node.DEBUG)
					System.out.print(" - "+valuesCount[j]+"/"+numValues + "*log(" +valuesCount[j] +"/"+numValues+")");
				
				tempSum -= (valuesCount[j]/numValues) * (Math.log((valuesCount[j]/numValues))/Math.log(2));
			}
			
			sum += firstTerm*tempSum;
			if (Node.DEBUG)
				System.out.println(")");			
			
		}
		if (Node.DEBUG) {
			System.out.println("= " + sum);
			System.out.println();
		}
		return sum;
	}
	
	public int getNumChildren() {
		if (this.children == null) {
			return 0; 
		}
		return this.children.length;
	}
	
	public String toString() {
		String labels = " Labels: {";
		for (int r = 0; r < this.dataSet[1].rows(); r++) {
			labels += (int) this.dataSet[1].get(r, 0) +",";
		}
		labels += "}";
		
		if (this.leafNode || this.dataSet == null || this.dataSet[0] == null) {
			return "Leaf Node. Data set size: " + this.dataSet[0].rows() + labels;
		}
		return "Split on: " + this.dataSet[0].getAttributeName(splitIndex) + ". Dataset size: "  + this.dataSet[0].rows() + labels;
	}
}
