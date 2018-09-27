package labs;
import java.util.Random;

import labs.Matrix;
import labs.Neuron;
import labs.SupervisedLearner;

public class Perceptron extends SupervisedLearner {
	private Neuron ner;
	private int maxEpoch;	
	private double[] accuracyList;
	private int lookBack;//number of epochs to compare accuracy to check for stall
	private double stalledAccuracy; //if accuracy doesn't improve by this much over <lookBack> epoch's then we've stalled
	private Random rand;
	
	//size is the number of inputs
	public Perceptron(Random rand) {			
		this.maxEpoch = 20;//max number of epoch's to train for
		this.accuracyList = new double[maxEpoch];//keep a list of accuracy of each epoch
		this.lookBack = 5;//we want to look back over 5 epoch's when checking for a stall
		this.stalledAccuracy = .01;//1%
		this.rand = rand;
	}
	
	public void trainThreeOutput(Matrix features, Matrix labels) throws Exception {
		//setup stuff
		Neuron[] n = new Neuron[3];
		for (int i = 0; i < 3; i++) {
			n[i] = new Neuron(this.rand, features.cols());
		}
		
		Matrix output = new Matrix();
		output.setSize(features.rows(), 1);
		
		for (int j = 0; j < this.maxEpoch; j++) {
			System.out.println("Epoch: " + (j+1));
			//for all rows
			for (int i = 0; i < features.rows(); i++) {
				//setup stuff to feed row into all three neurons
				double[] net = new double[3];
				double prediction = 0;
				double target = labels.get(i, 0);
				double greatestNet = 0;
				
				for (int k = 0; k < 3; k++) {//loop for all three inputs on this for
					net[k] = n[k].netOutput(features.row(i));//predict output
					
					if (k != (int) target && (net[k] > 0)) {//if the kth neuron fired when it wasn't supposed to
						n[k].objectiveFunction(0, 1);//update weights for a target of 0 and output of 1						
					}
					
					if (net[k] > greatestNet && net[k] > 0) {//pick output that has biggest net
						prediction = k;//0,1,2 
						greatestNet = net[k];//track biggest net for comparison
					}
				}
				System.out.println("prediction: " + prediction + " target: " + target);							
				output.set(i, 0, prediction);//track output
				
				for (int k = 0; k < 3; k++) {//loop through all neurons
					if (prediction != labels.get(i, 0)) {//if output was incorrect
						n[k].objectiveFunction(labels.get(i, 0), prediction);//update the weights
					}
				}
			}						
						
			double accuracy = this.accuracy(output, labels);
			this.accuracyList[j] = accuracy;//save the accuracy of this epoch
						
			System.out.println("accuracy: " + accuracy);					
			
			if (this.stalled(j)) {//if we have stalled in the amount of accuracy we can get				
				return;
			}
			System.out.println("");
		}
	}
	
	@Override
	//features is the data inputs
	//labels is the targets
	public void train(Matrix features, Matrix labels) throws Exception {
		//setup
		ner = new Neuron(this.rand, features.cols());
		Matrix output = new Matrix();
		output.setSize(features.rows(), 1);
		
		for (int j = 0; j < this.maxEpoch; j++) {			
			//for all rows
			for (int i = 0; i < features.rows(); i++) {
				double prediction = ner.output(features.row(i));//calculate the output				
				output.set(i, 0, prediction);
		
				if (prediction != labels.get(i, 0)) {//if output was incorrect
					ner.objectiveFunction(labels.get(i, 0), prediction);//update the weights
				}				
			}								
			double accuracy = this.accuracy(output, labels);
			this.accuracyList[j] = accuracy;//save the accuracy of this epoch			
			
			if (this.stalled(j)) {//if we have stalled in the amount of accuracy we can get				
				this.printStats(j+1);
				return;
			}
		}
		this.printStats(this.maxEpoch);
	}
	
	private void printMissClassification() {
		System.out.println("Miss classification for epoch:");
		for (int i = 0; i < this.accuracyList.length; i++) {
			System.out.println("   Epoch [" + (i+1) + "], " + (1-this.accuracyList[i]));
		}
	}
	
	private void printStats(int epochs) {
		System.out.println("Final weights: " + this.ner.getWeights());
		System.out.println("Final number of epochs: " + epochs);
		this.printMissClassification();
	}
	
	/**
	 * This function looks over past epoch's and if the greatest change in accuracy isn't bigger than this.stalledAccuracy
	 * The assume we have stalled
	 * @param epochNum number of epochs to look over
	 * @return
	 */
	private boolean stalled(int epochNum) {
		if (epochNum < this.lookBack) {//we have to have at least run for <lookBack> epochs to be able to lookBack
			return false;
		}
		double biggestDifference = 0;
		
		//loop over the past <lookBack> epochs
		//search for the biggest change in accuracy	
		for (int i = epochNum-1; i >= epochNum - this.lookBack; i--) {
			double diff = Math.abs(this.accuracyList[epochNum] - this.accuracyList[i]);			
			if (diff > biggestDifference) {
				biggestDifference = diff;
			}
		}
		//System.out.println("biggest diff: " + biggestDifference);
		if (biggestDifference < this.stalledAccuracy) {
			return true;
		}
		return false;
	}
	
	/** This function will return the accuracy of one pass through the data set
	 * @param output A vector of predicted outputs
	 * @param targets what the output was expected to be
	 * @return % of outputs that are correct
	 * @throws Exception 
	 */
	private double accuracy(Matrix output, Matrix targets) throws Exception {
		int correct=0, incorrect=0;
		
		if (output.rows() != targets.rows()) {
			throw (new Exception("Expected the features and labels to have the same number of rows"));
		}
		
		for (int i = 0; i < output.rows(); i++) {
			if (output.get(i, 0) == targets.get(i, 0)) {
				correct++;
			}
			else {
				incorrect++;
			}
		}
		
		return (double) correct / (double) (correct+incorrect);
	}

	@Override
	public void predict(double[] features, double[] labels) throws Exception {
		labels[0] = ner.output(features);
	}
	
	//A test function for testing the perceptron algorithm
	public static void main(String args[]) {
		Matrix m = new Matrix();		
		
		try {
			m.loadArff("iris.arff");
			m.shuffle(new Random());
			m.print();			
			
			int numCols = m.cols() - 1;//subtract one for target
			System.out.println("data columns: " + numCols);
			System.out.println(m.get(0, 4));
			
			Matrix targets = new Matrix(m, 0,numCols,m.rows(),1);			
			
			Matrix data = new Matrix(m,0,0,m.rows(),numCols);
			
			Perceptron per = new Perceptron(new Random());
			per.trainThreeOutput(data, targets);
			
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
