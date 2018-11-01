package decisionTree;

import java.io.FileNotFoundException;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

import labs.Matrix;
import labs.SupervisedLearner;

public class DecisionTree extends SupervisedLearner {
	private Node root;//the root node for this tree
	private static final double MISSING = Double.MAX_VALUE;
	
	@Override
	public void predict(double[] features, double[] labels) throws Exception {
		//handle missing values:
		for (int i = 0; i < features.length; i++) {
			if (features[i] == DecisionTree.MISSING) {
				features[i] = root.getData().mostCommonValue(i);
			}
		}
		
		Node n;		
		for (n = this.root; !n.isLeafNode();) {
			//this loop will go until n is a leaf node
			
			int splitIndex = n.getSplitIndex();//get the feature to split on			
			//System.out.print("Split index: " + splitIndex + ", ");
			int childIndex = (int) features[splitIndex];//get the index of the child
			
			//System.out.println("child index: " + childIndex);
			
			n = n.getChild(childIndex);
		}
		
		//System.out.println("Using node: " + n.toString() + " to predict output");
		Matrix outputLabels = n.getLabels();
		if (Node.DEBUG) {
			System.out.println("using the following matrix to predict output: ");
			outputLabels.print();
		}
		labels[0] = outputLabels.mostCommonValue(0);//get the majority of the output as the prediction	
		if (Node.DEBUG)
			System.out.println("Predicted value: " +  labels[0]);
	}
	
	public void train(Matrix features, Matrix labels) throws Exception {
		HashSet<Integer> usedAttributes = new HashSet<Integer>();
		root = new Node();
		System.out.println("Training");		
		
		//take care of missing values
		for (int r = 0; r < features.rows(); r++) {
			double[] row = features.row(r);
			
			for (int c = 0; c < row.length; c++) {				
				double averageValue = features.columnMean(c);//get average value of column
				if (features.get(r, c) == Matrix.MISSING) {//set missing values to the average value for that column
					features.set(r, c, averageValue);
				}
			}
		}
		
		//setting the root to have a dataset
		Matrix[] data = {features, labels};
		this.root.setDataSet(data);//set the node to have the incoming data
				
		this.MakeTree(root, features, labels, usedAttributes, 0);
		
		if (Node.DEBUG)
			this.printTree();
	}
	
	public void MakeTree(Node node, Matrix features, Matrix labels, HashSet<Integer> usedAttributes, int level) throws Exception {
		//variables to save the lowest index value
		int lowestIndex = Integer.MAX_VALUE;
		double lowestInfo = Double.MAX_VALUE;		
		if (node == null) {
			node = new Node();//create blank node
		}		
		
		//setting the node to have a dataset
		Matrix[] data = {features, labels};
		node.setDataSet(data);//set the node to have the incoming data
		
		if (Node.DEBUG) {
			System.out.println("Incoming features and labels: ");
			features.print();
			labels.print();
		}
		
		//get info for each attribute
		for (int attribute = 0; attribute < features.cols(); attribute++) {//for each attribute in the data set
			if (usedAttributes.contains(attribute)) {//if we have already split the data on this attribute
				continue;//skip this attribute
			}
			
			double info = node.info(features, attribute, labels);
			if (info < lowestInfo) {
				lowestIndex = attribute;
				lowestInfo = info;
			}
		}		
				
		
		if (labels.countUnique() == 1 || usedAttributes.size() >= features.cols() || level >= features.cols()) {//if we have reached a leaf node, or we have split on all attributes
			if (Node.DEBUG)
				System.out.println("This is a leaf node, no more need to split. The current level is: " + level + "*************************************\n");
			
			Matrix[] set = {features, labels};
			node.setDataSet(set);
			node.setLeaf();//set the node to be a leaf node
			return;
		}
		
		if (lowestIndex == Integer.MAX_VALUE) {
			System.out.println("got here");
		}
		
		HashSet<Double> values = (HashSet<Double>) features.getFeatureValues(features, lowestIndex);//get all unique values for this feature
		//node.setChildrenSize(values.size());//create children nodes = to the number of values the split column has
		node.setChildrenSize(features.valueCount(lowestIndex));//counts the number of possible values this feature can have
		node.setSplitIndex(lowestIndex);
		
		if (Node.DEBUG)
			System.out.println("Splitting on: " + features.getAttributeName(lowestIndex) + ", Info: " + lowestInfo + " Split index: " + lowestIndex + "\n");
		
		//finish building tree, by splitting data set on the best feature and sending it to each node
		//int index = 0;
		for (Iterator<Double> i = values.iterator(); i.hasNext();) {//for each unique value in best feature (column)
			double val = i.next();//get next unique value
			
			int numValues = features.countVal(lowestIndex, val);//count the number of instances that have this value
			
			//select all of the instances in features where the best column has a value of val
			Matrix split[] = features.select(features, lowestIndex, val, labels, numValues);//get a sub matrix all of the values of the best column are the same (equal to val)
//			if (Node.DEBUG) {
//				System.out.println("Split datasets:");
//				for (int j = 0; j < split.length; j++) {
//					split[j].print();
//				}
//			}
			//node.setDataSet(split);//save this split data set for later use
			node.setSplitIndex(lowestIndex);
			
			HashSet<Integer> used = new HashSet<Integer>(usedAttributes);//make a copy of the used attributes, so when we go back up the tree we don't have the used attributes from futher down
			used.add(lowestIndex);
			MakeTree(node.getChild((int)val), split[0], split[1], used, level+1);
		}
	}
	
	public void printTree() {
		System.out.println("0: " + this.root.toString());
		
		for (int i = 0; i < this.root.getNumChildren(); i++) {
			printTree(root.getChild(i), 1);
		}
	}
	
	public void printTree(Node node, int layer) {
		if (node == null) {
			return;
		}
		for (int i = 0; i < layer; i++) {
			System.out.print(" ");
		}
		System.out.print(layer + ": ");
		System.out.println(node.toString());
		
		for (int i = 0; i < node.getNumChildren(); i++) {
			printTree(node.getChild(i), layer+1);
		}
	}
	
	public static void main(String[] args) throws FileNotFoundException, Exception {
		//Node x = new Node();
		Matrix test = new Matrix();
		test.loadArff("id.arff");
		
		test.print();
		//System.out.println(x.probability(0, 0, test));
		
		Matrix features = new Matrix(test, 0, 0, test.rows(), test.cols()-1);
		Matrix labels = new Matrix(test, 0, test.cols()-1, test.rows(), 1);		
		
		DecisionTree tree = new DecisionTree();
		tree.train(features, labels);
		tree.printTree();
		
		double[] f = features.row(0);//get last instance
		double l[] = new double[1];
		
		tree.predict(f, l);
		
		System.out.println("output: " + l[0]); 
		
		/*
		int lowestIndex = 0;
		double lowestInfo = Double.MAX_VALUE;
		for (int i = 0; i < features.cols(); i++) {
			double info = x.info(features,i, labels); 
			if (info < lowestInfo) {
				lowestInfo = info;
				lowestIndex = i;
			}			
		}
		
		int numValues = features.countVal(lowestIndex, 0);			
		Matrix selection[] = features.select(features, lowestIndex, 0, labels, numValues);//sub[0] will have the selected features, sub[1] will have the labels to match
		selection[0].print();*/		
	}
}
