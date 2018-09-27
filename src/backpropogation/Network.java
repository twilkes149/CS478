package backpropogation;

import java.util.Random;

public class Network {
	private int numInputs;
	private int numLayers;
	private Node[][] layers;
	
	/**
	 * 
	 * @param inputs number of inputs
	 * @param layers each element defines how many nodes each layer has
	 */
	public Network(int inputs, int[] layers, Random r) {
		this.numInputs = inputs;
		this.numLayers = layers.length;
		this.layers = new Node[this.numLayers][];		
		
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
			}
		}
	}
	
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
	
	public static String arrayToString(double[] array) {
		String result = "[";
		for (int i = 0; i < array.length; i++) {
			result += array[i] + ",";
		}		
		return result + "]";
	}
	
	public static void main(String[] args) {
		Random rand = new Random(1);
		int[] layers = {10,5,5,5,5,6,7,8,9,4};//nodes per layer
		Network x = new Network(10, layers,rand);//10 layers
		
		double[] inputs = {0,1};
		System.out.println("Final output:" + Network.arrayToString(x.forwardPropogation(inputs)));
	}
}
