package decisionTree;

import java.io.FileNotFoundException;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

import labs.Matrix;

public class DecisionTree {
	private Node root;//the root node for this tree
	
	public void train(Matrix features, Matrix labels) throws Exception {
		HashSet<Integer> usedAttributes = new HashSet<Integer>();
		this.MakeTree(root, features, labels, usedAttributes);
	}
	
	public void MakeTree(Node node, Matrix features, Matrix labels, HashSet<Integer> usedAttributes) throws Exception {
		//variables to save the lowest index value
		int lowestIndex = 0;
		double lowestInfo = Double.MAX_VALUE;
		boolean leafNode = true;
		if (node == null) {
			node = new Node();//create blank node
		}		
		System.out.println("Incoming features and labels: ");
		features.print();
		labels.print();
		
		//get info for each attribute
		for (int attribute = 0; attribute < features.cols(); attribute++) {//for each attribute in the data set
			if (usedAttributes.contains(attribute)) {//if we have already split the data on this attribute
				continue;//skip this attribute
			}
			
			double info = node.info(features, attribute, labels);
			if (info > 0) {//if one of the attributes info is not zero, then this is not a leaf node
				leafNode = false;				
			}
			if (info < lowestInfo) {
				lowestIndex = attribute;
				lowestInfo = info;
			}
		}
		
		if (leafNode || usedAttributes.size() >= features.cols()) {//if we have reached a leaf node, or we have split on all attributes
			System.out.println("This is a leaf node, no more need to split\n");
			return;
		}
		
		HashSet<Double> values = (HashSet<Double>) features.getFeatureValues(features, lowestIndex);//get all unique values for this feature
		node.setChildrenSize(values.size());//create children nodes = to the number of values
		
		System.out.println("Splitting on: " + features.getAttributeName(lowestIndex) + ", Info: " + lowestInfo + "\n");
		
		//finish building tree, by splitting data set on the best feature and sending it to each node
		int index = 0;
		for (Iterator<Double> i = values.iterator(); i.hasNext(); index++) {//for each unique value in best feature
			double val = i.next();
			int numValues = features.countVal(lowestIndex, val);//count the number of instances that have this value
			
			Matrix split[] = features.select(features, lowestIndex, val, labels, numValues);//get a sub matrix were the best feature is split into n matrices were n is the number of unique values
			
			HashSet<Integer> used = new HashSet<Integer>(usedAttributes);//make a copy of the used attributes, so when we go back up the tree we don't have the used attributes from futher down
			used.add(lowestIndex);
			MakeTree(node.getChild(index), split[0], split[1], used);
		}
	}
	
	public static void main(String[] args) throws FileNotFoundException, Exception {
		Node x = new Node();
		Matrix test = new Matrix();
		test.loadArff("id.arff");
		
		test.print();
		//System.out.println(x.probability(0, 0, test));
		
		Matrix features = new Matrix(test, 0, 0, test.rows(), test.cols()-1);
		Matrix labels = new Matrix(test, 0, test.cols()-1, test.rows(), 1);		
		
		DecisionTree tree = new DecisionTree();
		tree.train(features, labels);
		
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
