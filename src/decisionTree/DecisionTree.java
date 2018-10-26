package decisionTree;

import java.io.FileNotFoundException;

import labs.Matrix;

public class DecisionTree {
	public void MakeTree(Node node, Matrix features, Matrix labels, int attributeIndex) throws Exception {
		//variables to save the lowest index value
		int lowestIndex = 0;
		double lowestInfo = Double.MAX_VALUE;
		boolean leafNode = true;
		
		for (int attribute = 0; attribute < features.cols(); attribute++) {//for each attribute in the data set
			double info = node.info(features, attribute, labels);
			if (info > 0) {//if one of the attributes info is not zero, then this is not a leaf node
				leafNode = false;
			}
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
		
		//features.print();
		//labels.print();
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
		selection[0].print();		
	}
}
