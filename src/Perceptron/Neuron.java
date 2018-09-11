package Perceptron;

import java.util.ArrayList;
import java.util.Random;

public class Neuron {
	private ArrayList<Float> weights;//includes all weights except the bias
	private ArrayList<Float> inputs;
	
	private float learningRate;
	private float biasWeight;
	private float output;
	public float net;
	
	Neuron(int size) {
		this.weights = new ArrayList<Float>();
		this.learningRate = (float) 0.1;
		
		//init all weights to random numbers
		Random rand = new Random();
		for (int i = 0; i < size; i++) {
			weights.add(rand.nextFloat());//get a random number between 0 and 1
		}
	}

	public void objectiveFunction(float target, float output, float expected, int index) {
		//return learning rate * (target - output) * input[index]
		if (output != expected) {//if output didn't match our target, update weight
			this.weights.set(index, this.weights.get(index) +
				this.learningRate * (target - output)*inputs.get(index));
		}
	}
	
	//returns sum of linear combination of weights and inputs
	public Float netOutput(ArrayList<Float> inputs_) {
		this.inputs = inputs_;
		float sum = 0;
		
		for (int i = 0; i < inputs_.size(); i++) {
			inputs_.set(i,inputs_.get(i)*weights.get(i));
			sum += inputs_.get(i);
		}
		sum += this.biasWeight*1;//adding the bias on
		this.net = sum;//save the output for later
		return sum;		
	}
	
	//runs the netOutput function
	//calculates the output of this neuron
	public boolean output(ArrayList<Float> inputs_) {
		this.netOutput(inputs_);
		if (this.net > 0) {
			this.output = 1;
			return true;
		}
		this.output = 0;
		return false;
	}
	
	public String toString() {
		return "Weights: " + this.weights.toString();
	}
	
	public static void main(String args[]) {
		Neuron x = new Neuron(2);
		ArrayList<Float> inputs = new ArrayList<Float>();
		inputs.add((float) 1);
		inputs.add((float) 2);
		
		System.out.println(x.toString());
		System.out.println(x.output(inputs) + " " + x.net);
	}
}
