package backpropogation;

import java.util.Random;

import labs.Matrix;

public class Network {
	private int numInputs;
	private int numLayers;
	private double learningRate;
	private double[] inputs;
	private Node[][] layers;
	private static final boolean DEBUG = false;
	
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
		this.inputs = inputs;//save for later reference in backprop
		//runs all of the inputs through the network and calculates the output
		double[] layerInput = inputs;
		double[] layerOutput;
		for (int i = 0; i < this.numLayers; i++) {//for each layer			
			layerOutput = new double[this.layers[i].length];//array to hold output for each layer
			
			for (int j = 0; j < this.layers[i].length; j++) {//for each node in layer
				layerOutput[j] = this.layers[i][j].calcOutput(layerInput);//get output for this node
			}
			//if (i == this.numLayers-1) {
				//System.out.println("Layer["+i+"] output: " + Network.arrayToString(layerOutput));
			//}			
			layerInput = layerOutput;//set output of this layer as input to next layer
			//System.out.println("Layer["+i+"] output:" + Network.arrayToString(layerOutput));
		}
		
		return layerInput;//finally return last layer output
	}
	
	public void backPropogation(double[] targets) throws Exception {
		//calculate output nodes
		int outputIndex = this.layers.length-1 < 0 ? 0 : this.layers.length-1;
		
		for (int i = 0; i < this.layers[outputIndex].length; i++) {//loop through all output nodes			
			double error = this.layers[outputIndex][i].calcError(targets[i]);//calculating the error of the output
			
			if (Network.DEBUG) {
				//System.out.println("  Updating weights for node: " + i);
				//System.out.println("    Error: " + error);
			}
			for (int j = 0; j < this.layers[outputIndex][i].numWeights(); j++) {//for each weight of output node
				//i is the node, j is the weight of the node
				double prevNodeOutput = this.layers[outputIndex-1][j].getOutput();//get the output of the jth node on the previous layer
				if (Network.DEBUG) {
					System.out.println("prevLayer: " + (outputIndex-1) + " node: " + j + ", output: " + prevNodeOutput);
				}
				this.layers[outputIndex][i].updateWeight(j, error, prevNodeOutput);
				if (Network.DEBUG) {
					//System.out.println("    Delta W[output][" + i + "][" + j +"]: " + this.learningRate*error*prevNodeOutput);
				}
			}
			//update bias weight
			int biasWeightIndex = this.layers[outputIndex][i].numWeights();
			this.layers[outputIndex][i].updateWeight(biasWeightIndex, error, 1);
		}
		
		if (Network.DEBUG) {
			for (int i = 0; i < this.layers[outputIndex].length; i++) {//loop through all output layer nodes
				System.out.println("New output layer weights: " + Network.arrayToString(this.layers[outputIndex][i].getWeights()));//print out their weights
			}
		}
		
		//middle layer nodes
		for (int layerIndex = outputIndex-1; layerIndex >= 0; layerIndex--) {//for each layer
			if (Network.DEBUG) {
				System.out.println("Updating layer[" + layerIndex + "]:");
			}
			for (int nodeIndex = 0; nodeIndex < this.layers[layerIndex].length; nodeIndex++) {//for each node on layer
				if (Network.DEBUG) {
					//System.out.println("  Node[" + nodeIndex + "]:");
				}
				
				//calculate error of middle node
				double error = 0;
				if (Network.DEBUG) {
					//System.out.println("    Next layer length: " + this.layers[layerIndex+1].length);
				}
				for (int nextLayerNode = 0; nextLayerNode < this.layers[layerIndex+1].length; nextLayerNode++) {//loop for each node in the next layer
					if (Network.DEBUG) {
						//System.out.println("      NL_Node[" + nextLayerNode + "]:");
					}
					error += this.layers[layerIndex+1][nextLayerNode].getError() * this.layers[layerIndex+1][nextLayerNode].getWeight(nodeIndex);//error of next node * weight connecting these two nodes
				}
				error *= this.layers[layerIndex][nodeIndex].sigmoidPrime(this.layers[layerIndex][nodeIndex].getNet());
				this.layers[layerIndex][nodeIndex].setError(error);//set the error of this node to use later
				if (Network.DEBUG) {
					System.out.println("    Error: " + error);
				}
				
				//update weights
				for (int weightIndex = 0; weightIndex < this.layers[layerIndex][nodeIndex].numWeights(); weightIndex++) {//for all weights
					double prevNodeOutput=0;
					if (layerIndex == 0) {//first layer
						prevNodeOutput = this.inputs[weightIndex];//the prevNode was an input
						if (Network.DEBUG) {
							System.out.println("prevLayer: input " + " node: " + weightIndex + ", output: " + prevNodeOutput);
						}
					}
					else {
						prevNodeOutput = this.layers[layerIndex-1][weightIndex].getOutput();//grab previous layer node output
						if (Network.DEBUG) {
							System.out.println("prevLayer: " + (layerIndex-1) + " node: " + weightIndex + ", output: " + prevNodeOutput);
						}
					}						
					this.layers[layerIndex][nodeIndex].updateWeight(weightIndex, error, prevNodeOutput);
				}
				//update bias weight
				int biasWeightIndex = this.layers[layerIndex][nodeIndex].numWeights();
				this.layers[layerIndex][nodeIndex].updateWeight(biasWeightIndex, error, 1);
				
				if (Network.DEBUG) {
					System.out.println("    New weights: " + Network.arrayToString(this.layers[layerIndex][nodeIndex].getWeights()));
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
			result += "Layer[" + layers + "] (" + this.layers[layers].length + " nodes):\n";
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
		System.out.println("Network: " + x.toString());
		
		double[] inputs = {0,0};
		System.out.println("Final output:" + Network.arrayToString(x.forwardPropogation(inputs)));		
		double[] targets = {1};
		try {
			x.backPropogation(targets);
		
			inputs[1] = 1;//new input
			targets[0] = 0;
			System.out.println("Final output:" + Network.arrayToString(x.forwardPropogation(inputs)) + "\n\n\n");
			x.backPropogation(targets);
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
}
