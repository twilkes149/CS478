package KNN;

import backpropogation.Network;
import decisionTree.Node;
import labs.Matrix;
import labs.SupervisedLearner;

public class NearestNeighbor extends SupervisedLearner {
	private boolean useDistanceWeighting; //whether or not to use distance weighting
	private int k;//k nearest neighbors
	private Neighbor[] neighbors;//a list of the k nearest neighbors
	private int freq; //used if we are only keeping a subset of the training features	
	Matrix features, labels; //used to save the dataset
	
	public static final boolean DEBUG = false;
	public static final boolean LOOP_DEBUG = false;
	public static final boolean DEBUG_VOTING = false;
	
	public NearestNeighbor() {
		this.useDistanceWeighting = true;
		this.k = 3;
		this.neighbors = new Neighbor[this.k];//a list of the k nearest neighbors
		this.freq = 1;//keep every instance in features		
	}

	@Override
	public void train(Matrix features, Matrix labels) throws Exception {		
		this.features = features;
		this.labels = labels;		
	}

	@Override
	public void predict(double[] features, double[] labels) throws Exception {//this gets an instance of our test set
		int numberOfOutputClasses = this.labels.countUnique();
		//reset neighbors array
		for (int i = 0; i < this.k; i++) {
			this.neighbors[i] = new Neighbor();
		}
		
		for (int i = 0; i < this.features.rows(); i++) {//for every instance in the training set
			double distance = 0;
			for (int col = 0; col < this.features.cols(); col++) {//calculate the distance from this instance
				distance += Math.abs(features[col] - this.features.get(i, col));
			}
			if (NearestNeighbor.LOOP_DEBUG) {								
				if (distance == 0) {
					System.out.println(i + ": Distance: " + distance + ", label: " + this.labels.get(i, 0));
					System.out.println("Training: " + Network.arrayToString(this.features.row(i)) + "; test feature: " + Network.arrayToString(features));
				}
			}
			
			
			double farthestDistance = this.neighbors[0].getDistance();
			int farthestIndex = 0;
			
			for (int j = 1; j < this.neighbors.length; j++) {//get farthest neighbor out of the closest neighbors
				if (this.neighbors[j].getDistance() > farthestDistance) {
					
					farthestDistance = this.neighbors[j].getDistance();
					farthestIndex = j;
				}
			}			
			
			if (distance < farthestDistance && (distance != 0)) {//if this instance is closer than the farthest away nearest neighbor, override the nearest neighbor
				if (NearestNeighbor.LOOP_DEBUG) {
					System.out.println("Saved new nearest neighbor: " + farthestDistance + ", " + (int) this.labels.get(i,0));
				}
				//save the new closest neighbor
				this.neighbors[farthestIndex].setDistance(distance);
				this.neighbors[farthestIndex].setLabel((int) this.labels.get(i,0));
			}
		}
		
		if (NearestNeighbor.DEBUG_VOTING) {
			System.out.print("Neighbors: [");
			for (int i = 0; i < this.k; i++) {
				System.out.print(this.neighbors[i].toString());
			}
			System.out.println("]");
		}
		
		if (NearestNeighbor.DEBUG && false) {						
			for (int l = 0; l < this.k; l++) {
				System.out.println("ith neighbor distance: " + this.neighbors[l].getDistance() + ", label: " + this.neighbors[l].getLabel());
			}
		}
		
		double outputClasses[] = new double[numberOfOutputClasses];//an array to hold all of the votes for each class
		
		//find the class that has the most votes
		if (!this.useDistanceWeighting) {
			for (int i = 0; i < this.k; i++) {
				outputClasses[this.neighbors[i].getLabel()] += 1;//add a vote
			}
		}
		else {
			double denominator = 0, numerator = 0;
			for (int j = 0; j < this.k; j++) {//for each neighbor
				denominator += 1 / (this.neighbors[j].getDistance());
			}
			
			
			for (int i = 0; i < numberOfOutputClasses; i++) {//for each output class
				numerator = 0;
				for (int j = 0; j < this.k; j++) {//for each neighbor
					if (this.neighbors[j].getLabel() == i) {//if the neighbor's output class == the current class we are looking at
						numerator += 1/(this.neighbors[j].getDistance());
					}
				}
				
				outputClasses[i] += numerator/denominator;
			}
		}
		
		//count the class that has the most votes
		double mostVotes = 0, indexClass = 0;
		for (int i = 0; i < outputClasses.length; i++) {
			if (outputClasses[i] > mostVotes) {
				mostVotes = (int) outputClasses[i];
				indexClass = i;
			}
		}
		if (NearestNeighbor.DEBUG) {
			System.out.println("Predicting class: " + indexClass);
		}
		labels[0] = indexClass;//predict the class
	}
}
