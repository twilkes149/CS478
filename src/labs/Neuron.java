package labs;

import java.util.Random;

public class Neuron {
	private double[] weights;//includes all weights except the bias
	private double[] inputs;
	
	private double learningRate;
	private double biasWeight;
	private double output;
	public double net;
	
	//size is the number of inputs
	Neuron(Random rand, int size) {
		this.learningRate = (double) 0.1;
		//setup stuff		
		this.weights = new double[size];				
		
		//init all weights to random numbers		
		for (int i = 0; i < size; i++) {			
			weights[i] = rand.nextDouble();//get a random number between 0 and 1
		}
		biasWeight = rand.nextDouble();
	}
	
	public String getWeights() {
		String result = "";
		for (int i = 0; i < this.weights.length; i++) {
			result += this.weights[i] + " ";
		}
		return result += this.biasWeight;
	}

	public void objectiveFunction(double target, double output) {
		//return learning rate * (target - output) * input[index]
		for (int i = 0; i < this.inputs.length; i++) {
			this.weights[i] = this.weights[i] +
				this.learningRate * (target - output)*inputs[i];
		}
		this.biasWeight += this.learningRate*(target - output)*1;
	}
	
	//returns sum of linear combination of weights and inputs
	public Double netOutput(double[] inputs_) {
		this.inputs = inputs_;		
		double sum = 0;
		
		for (int i = 0; i < inputs_.length; i++) {			
			sum += inputs_[i]*weights[i];
		}
		sum += this.biasWeight*1;//adding the bias on
		this.net = sum;//save the output for later		
		return sum;		
	}
	
	//runs the netOutput function
	//calculates the output of this neuron
	public double output(double[] inputs_) {				
		this.netOutput(inputs_);
		if (this.net > 0) {
			this.output = 1;
			return (double) 1;
		}
		this.output = 0;
		return (double) 0;
	}
	
	public String toString() {
		String result =  "Weights: [";
		for (int i = 0; i < this.weights.length; i++) {
			result += this.weights[i] + ", ";
		}
		return result + "]";
	}
	
	//a simple test function 
	public static void main(String args[]) {
		Random rand = new Random();
		Neuron x = new Neuron(rand,2);
		double[] inputs = {1,2};
		
		System.out.println(x.output(inputs));
		System.out.println(x.toString());		
	}
}
