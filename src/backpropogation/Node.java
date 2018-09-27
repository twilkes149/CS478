package backpropogation;

import java.util.Random;

import labs.Matrix;

public class Node {
	private double[] weights;
	private double[] inputs;
	private double output;
	private double net;
	private Random rand;
	private double learningRate;
	private int numInputs;
	
	public Node(int numInputs, Random r) {
		this.weights = new double[numInputs+1];
		this.numInputs = numInputs;
		this.setWeights(1);
	}
	
	public void setLearningRate(double rate) {
		this.learningRate = rate;
	}
	
	//sets all the weights to a random number
	private void setWeights(Random r) {
		this.rand = r;
		//set all weights
		for (int i = 0; i < this.numInputs; i++) {
			this.weights[i] = this.rand.nextDouble();//set random weight
		}
		this.weights[this.numInputs] = 1;//set bias weight
	}
	
	//sets all the weights to a z
	private void setWeights(double z) {
		//set all weights
		for (int i = 0; i < this.numInputs; i++) {
			this.weights[i] = z;//set weight to z
		}
		this.weights[this.numInputs] = 1;//set bias weight
	}
	
	public double getOutput() {
		return this.output;
	}
	
	//the activation function
	private double sigmoid(double net) {
		return 1/(1+Math.exp(-1*net)); 
	}
	
	//the derivative of the sigmoid function
	private double sigmoidPrime(double net) {
		return this.sigmoid(net) * (1 - this.sigmoid(net));
	}
	
	//calculates linear combination of inputs and weights
	public double calcNet(double[] inputs_) {
		this.inputs = inputs_;
		this.net = 0;//init net
		
		for (int i = 0; i < this.inputs.length; i++) {//create linear combination of weights and inputs
			this.net += this.inputs[i] * this.weights[i];
		}
		this.net += 1*this.weights[this.numInputs];//adding bias
		return this.net;
	}
	
	//calculates net, and then feeds that into the sigmoid function to get output
	public double calcOutput(double[] inputs_) {
		double net = this.calcNet(inputs_);
		this.output = this.sigmoid(net);
		return this.output;
	}
	
	public void updateWeight(int index, double output, double error) {
		this.weights[index] = this.learningRate*output*error;
	}
}
