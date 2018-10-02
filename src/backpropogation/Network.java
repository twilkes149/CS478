package backpropogation;

import java.util.Random;

public class Network {
	private int numInputs;
	private int numLayers;
	private double learningRate;
	private Node[][] layers;
	private static final boolean DEBUG = true;
	
	/**
	 * 
	 * @param inputs number of inputs
	 * @param layers each element defines how many nodes each layer has
	 */
	public Network(int inputs, int[] layers, Random r, double learningRate) {
		this.numInputs = inputs;
		this.numLayers = layers.length;
		this.layers = new Node[this.numLayers][];
		this.learningRate = learningRate;
		
		//set up network
		for (int i = 0; i < layers.length; i++) {//for all layers
			this.layers[i] = new Node[layers[i]];//and array of nodes for this layer
			
			for (int j = 0; j < layers[i]; j++) {//set up each node for this layer
				if (i == 0) {
					this.layers[i][j] = new Node(this.numInputs,r);//setting up each node of the first layer to have this.inputs inputs
				}
				else {
					this.layers[i][j] = new Node(layers[i-1],r);//setting up a middle layer nodes to each have as many inputs as the previous layer had nodes
				}
				this.layers[i][j].setLearningRate(this.learningRate);
			}
		}
	}
	
	//function to propogate the inputs through the network and get the final output
	public double[] forwardPropogation(double[] inputs) {
		//runs all of the inputs through the network and calculates the output
		double[] layerInput = inputs;
		for (int i = 0; i < this.numLayers; i++) {//for each layer			
			double[] layerOutput = new double[this.layers[i].length];//array to hold output for each layer
			
			for (int j = 0; j < this.layers[i].length; j++) {//for each node in layer
				layerOutput[j] = this.layers[i][j].calcOutput(layerInput);//get output for this node
			}
			System.out.println("Layer["+i+"] output: " + Network.arrayToString(layerOutput));
			layerInput = layerOutput;//set output of this layer as input to next layer
		}
		
		return layerInput;//finally return last layer output
	}
	
	public void backPropogation(double[] targets) throws Exception {
		//calculate output nodes
		int outputIndex = this.layers.length-1;
		
		if (Network.DEBUG) {//if we are debugging
			System.out.println("Output layer index: " + outputIndex);
		}
		for (int i = 0; i < this.layers[outputIndex].length; i++) {//loop through all output nodes			
			double error = this.layers[outputIndex][i].calcError(targets[i]);//calculating the error of the output
			
			if (Network.DEBUG) {
				System.out.println("  Updating weights for node: " + i);
				System.out.println("    Error: " + error);
			}
			for (int j = 0; j < this.layers[outputIndex][i].numWeights(); j++) {//for each weight of output node
				//i is the node, j is the weight of the node
				double prevNodeOutput = this.layers[outputIndex-1][j].getOutput();//get the output of the jth node on the previous layer
				this.layers[outputIndex][i].updateWeight(j, error, prevNodeOutput);
				if (Network.DEBUG) {
					System.out.println("    Delta W[output][" + i + "][" + j +"]: " + this.learningRate*error*prevNodeOutput);
				}
			}
			//update bias weight
			int biasWeightIndex = this.layers[outputIndex][i].numWeights();
			this.layers[outputIndex][i].updateWeight(biasWeightIndex, error, 1);
		}
		System.out.println("Finished updating output layer weights");
		if (Network.DEBUG) {
			for (int i = 0; i < this.layers[outputIndex].length; i++) {//loop through all output layer nodes
				System.out.println("New output layer weights: " + Network.arrayToString(this.layers[outputIndex][i].getWeights()));//print out their weights
			}
		}
		
		//middle layer nodes
		for (int layerIndex = outputIndex-1; layerIndex >= 0; layerIndex--) {//for each layer
			for (int nodeIndex = 0; nodeIndex < this.layers[layerIndex].length; nodeIndex++) {//for each node on layer
				//calculate error of middle node
				double error = 0;
				for (int nextLayerNode = 0; nextLayerNode <= this.layers[layerIndex+1].length; nextLayerNode++) {//loop for each node in the next layer
					error += this.layers[layerIndex+1][nextLayerNode].getError() * this.layers[layerIndex+1][nextLayerNode].getWeight(nodeIndex);//error of next node * weight connecting these two nodes
				}
				this.layers[layerIndex][nodeIndex].setError(error);//set the error of this node to use later
				
				//update weights
				for (int weightIndex = 0; weightIndex < this.layers[layerIndex][nodeIndex].numWeights(); weightIndex++) {//for all weights
					double prevNodeOutput = this.layers[layerIndex-1][weightIndex].getOutput();
					this.layers[layerIndex][nodeIndex].updateWeight(weightIndex, error, prevNodeOutput);
				}
			}
		}
	}
	
	public static String arrayToString(double[] array) {
		String result = "[";
		for (int i = 0; i < array.length; i++) {
			result += array[i] + ",";
		}		
		return result + "]";
	}
	
	//returns a string representation of this network
	public String toString() {
		String result = "";
		for (int layers = 0; layers < this.layers.length; layers++) {
			result += "Layer[" + layers + "]:\n";
			for (int nodes = 0; nodes < this.layers[layers].length; nodes++) {
				result += "  node[" + nodes + "]:\n";
				result += "    weights:" + Network.arrayToString(this.layers[layers][nodes].getWeights()) + "\n";				
			}
		}
		
		return result;
	}
	
	public static void main(String[] args) {
		Random rand = new Random(1);
		int[] layers = {2,1};//nodes per layer
		Network x = new Network(2, layers,rand,1);//10 layers
		
		double[] inputs = {0,0};
		System.out.println("Final output:" + Network.arrayToString(x.forwardPropogation(inputs)));
		System.out.println("Network: " + x.toString());
		double[] targets = {1};
		try {
			x.backPropogation(targets);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
