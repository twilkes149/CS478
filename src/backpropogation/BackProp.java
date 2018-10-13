package backpropogation;

import java.util.Random;

import labs.Matrix;
import labs.SupervisedLearner;

//TODO I don't think the accuracy function is working 

public class BackProp extends SupervisedLearner {
	private int maxEpoch;	
	private double[] accuracyList;//used to track the accuracy of each epoch
	private int lookBack;//number of epochs to compare accuracy to check for stall
	private double stalledAccuracy; //if accuracy doesn't improve by this much over <lookBack> epoch's then we've stalled
	private double validationSetPercent;
	private Random rand;	
	private double learningRate;
	
	private static final boolean DEBUG = false;
	
	private Network network;
	private int layers;
	private int[] nodesPerLayer;
	private int outputClasses;
	
	public BackProp(Random r) {
		this.rand = r;
		this.maxEpoch = 80;
		this.lookBack = 5;
		this.stalledAccuracy = .005;
		this.accuracyList = new double[maxEpoch];//keep a list of accuracy of each epoch
		this.layers = 3;	
		this.learningRate = .1;
		this.validationSetPercent = .3;
		this.network = null;
		
		this.setNodesPerLayer(null);
	}
	
	public String toString() {
		return this.network.toString();
	}
	
	/**
	 * This function defines the structure of the network. It must be called before training the network
	 * if the nodes parameter is null, a default network will be created
	 * The first layer must have as many nodes as input features
	 * @param nodes an array where the nth element defines how many nodes the nth layer has
	 */
	public void setNodesPerLayer(int[] nodes) {
		if (nodes == null) {
			this.nodesPerLayer = new int[3];//default network
			this.nodesPerLayer[0] = 3;
			this.nodesPerLayer[1] = 2;
			this.nodesPerLayer[2] = 1;
		}
		else {
			this.nodesPerLayer = nodes;
		}
	}
	
	public void setNodesPerLayer(int numInputs) {
		this.nodesPerLayer = new int[2];
		this.nodesPerLayer[0] = numInputs*2;
//		this.nodesPerLayer[1] = numInputs*2;
		this.nodesPerLayer[1] = this.outputClasses;
	}
	
	/**
	 * This function turns an output class into a oneHot encoding array
	 * @param targetClass The number representing the output class
	 * @return one hot encoding of the class i.e class 3, in a 5 output-class data-set turns into [0,0,0,1,0]
	 */
	private double[] oneHot(double targetClass) {
		double[] oneHot = new double[this.outputClasses];
		for (int i = 0; i < this.outputClasses; i++) {
			if (i == (int) targetClass) {
				oneHot[i] = 1;
			}
			else {
				oneHot[i] = 0;
			}
		}
		return oneHot;
	}
	
	/**
	 * Returns an output class (int) based on the highest output
	 * @param output
	 * @return
	 */
	public int interpretOutput(double[] output) {
		double greatest=0;
		int index=0;
		
		for (int i = 0; i < output.length; i++) {
			if (output[i] > greatest) {
				index = i;
				greatest = output[i];//save biggest output
			}
		}
		return index;//return the output class
	}

	@Override
	public void train(Matrix features, Matrix labels) throws Exception {
		//grab validation set
		int vSetSize = (int) (features.rows()*this.validationSetPercent);
		int tSetSize = (int) features.rows()-vSetSize;
		
		Matrix validation = new Matrix(features, 0, 0,vSetSize,features.cols());
		Matrix validateLabels = new Matrix(labels, 0, 0,vSetSize,labels.cols());
		
		Matrix test = new Matrix(features,vSetSize,0,tSetSize,features.cols());
		Matrix testLabels = new Matrix(labels,vSetSize,0,tSetSize,labels.cols());
		
		//set up stuff	
		Matrix output = new Matrix();
		output.setSize(test.rows(), 1);
		this.outputClasses = testLabels.countUnique();
		this.setNodesPerLayer(test.cols());
				
		System.out.println("Output classes: " + this.outputClasses);
		
		this.network = new Network(test.cols(), this.nodesPerLayer, this.rand, this.learningRate);//create the network
		
		for (int epoch = 0; epoch < this.maxEpoch; epoch++) {//at most go MAX_EPOCH number of times
			//if (BackProp.DEBUG) {
				//System.out.println("Epoch["+epoch+"]:");
			//}
			for (int instance = 0; instance < test.rows(); instance++) {//for all data instances				
				double[] prediction = this.network.forwardPropogation(test.row(instance));
				if (BackProp.DEBUG) {
					System.out.println("Input: " + Network.arrayToString(test.row(instance)));
					System.out.println("Output: " + Network.arrayToString(prediction));
					System.out.println("Finished forward prop");
				}
				
				double outputClass = this.interpretOutput(prediction);//get one hot encoding of prediction
				double targetClass = testLabels.get(instance, 0);//get target class
				
				//System.out.println("output: " + outputClass + " target: " + targetClass + " prediction: " + Network.arrayToString(prediction));
				
				output.set(instance, 0, outputClass);//save the output of this instance for use in determining accuracy
				double[] oneHot = this.oneHot(targetClass);//get one hot encoding for target class
				
				if (BackProp.DEBUG) {
					System.out.println("Target class: " + targetClass);
					System.out.println("One hot target: " + Network.arrayToString(oneHot));
				}
				
				this.network.backPropogation(oneHot);//update weights, based on target
			}//end for all instances
			
			double accuracy = this.accuracy(output, testLabels);
			System.out.println("Epoch["+epoch+"]: Accuracy: " + accuracy);
			this.accuracyList[epoch] = accuracy;//save the accuracy of this epoch
			
			if (this.stalled(epoch)) {
				System.out.println("Accuracy has stopped improving at epoch: " + (epoch+1));
				return;
			}
			if (BackProp.DEBUG) {
				System.out.println("End epoch\n\n");
			}
			
			test.shuffle(this.rand, testLabels);//randomize features after each epoch
		}	//end for all epochs
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
	/**
	 * This network must be trained before this method can be called
	 */
	public void predict(double[] features, double[] labels) throws Exception {
		if (this.network == null) {
			throw new Exception("Network hasn't been trained yet");
		}
		else {
			double[] prediction = this.network.forwardPropogation(features);
			labels[0] = this.interpretOutput(prediction);//get one hot encoding of prediction
			
		}
		
	}
	
}
