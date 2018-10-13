package backpropogation;

import java.util.Random;

import labs.Matrix;

public class Node {
	private double[] weights;
	private double[] inputs;
	private double output;
	private double error;
	private double net;
	private Random rand;
	private double learningRate;
	private int numInputs;
	private double momentum;
	
	private static final boolean useMomentum = false;//set to true if we use momentum, set to false if we don't want to use momentum
	
	public Node(int numInputs, Random r) {
		this.weights = new double[numInputs+1];
		this.numInputs = numInputs;
		this.setWeights(r);//set all weights to 1
		
		this.momentum = .00005;
	}
	
	public void setLearningRate(double rate) {
		this.learningRate = rate;
	}
	
	//sets all the weights to a random number
	private void setWeights(Random r) {
		this.rand = r;
		//set all weights
		for (int i = 0; i < this.numInputs; i++) {
			this.weights[i] = this.rand.nextDouble()/10;//set random weight, divide by 10 to get a small number
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
	
	public double getError() {
		return this.error;
	}
	
	public double getNet() {
		return this.net;
	}
	
	public double getWeight(int index) {
		return this.weights[index];
	}
	
	public double[] getWeights() {
		return this.weights;
	}
	
	public int numWeights() {
		return this.weights.length-1;//compensate for bias weight
	}
	
	//the activation function
	public double sigmoid(double net) {
		return 1/(1+Math.exp(-1*net)); 
	}
	
	//the derivative of the sigmoid function
	public double sigmoidPrime(double net) {
		return this.sigmoid(net) * (1 - this.sigmoid(net));
	}
	
	//calculates linear combination of inputs and weights
	public double calcNet(double[] inputs_) {
		this.inputs = inputs_;
		this.net = 0;//init net
		
		for (int i = 0; i < this.inputs.length; i++) {//create linear combination of weights and inputs
			this.net += (this.inputs[i] * this.weights[i]);
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
	
	//calculate the error of this node if it is an output node 
	public double calcError(double target) {
		this.error = (target - this.output)*this.sigmoidPrime(this.net);
		return this.error;
	}
	
	public void setError(double e) {
		this.error = e;
	}
	
	//calculate the error or this node if it's a middle layer node
//	public double calcError(double[] nextNodeError, double[] weights) throws Exception {
//		if (nextNodeError.length != weights.length) {
//			throw new Exception("Erros and weights must be same length");
//		}
//		this.error = this.sigmoidPrime(this.net);
//		for (int i = 0; i < weights.length; i++) {
//			this.error *= weights[i]*nextNodeError[i];
//		}
//		
//		return this.error;
//	}
	
	/**
	 * Updates one weight in this node
	 * @param index the index of the weight to update
	 * @param error the error of this node
	 * @param prevNodeOutput the output of a previous layer node, connected through this weight
	 */
	public void updateWeight(int index, double error, double prevNodeOutput) {
		if (Node.useMomentum) {
			this.weights[index] += this.learningRate*prevNodeOutput*error + this.momentum*this.weights[index];//update the weight
		}
		else {
			this.weights[index] += this.learningRate*prevNodeOutput*error;//update the weight
		}
	}
}
