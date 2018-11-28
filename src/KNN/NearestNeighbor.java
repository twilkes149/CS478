package KNN;

import java.util.HashSet;
import java.util.Set;

import backpropogation.Network;
import decisionTree.Node;
import labs.Matrix;
import labs.SupervisedLearner;

public class NearestNeighbor extends SupervisedLearner {
	private boolean useDistanceWeighting; //whether or not to use distance weighting
	private boolean regression;//whether or not to use knn regression
	private boolean hetogeneous;//used to tell the algorithm whether the data set is all linear features or if its a mix between linear and nominal
	private int k;//k nearest neighbors
	private Neighbor[] neighbors;//a list of the k nearest neighbors
	private int freq; //used if we are only keeping a subset of the training features	
	Matrix features, labels; //used to save the dataset
	
	Set<Integer> skip;

	
	public static final boolean DEBUG = false;
	public static final boolean DISTANCE_DEBUG = false;
	public static final boolean LOOP_DEBUG = false;
	public static final boolean DEBUG_VOTING = false;
	public static final boolean REGRESSION_DEBUG = false;
	
	public NearestNeighbor() {
		this.useDistanceWeighting = false;
		this.hetogeneous = false;
		this.regression = false;
		this.k = 5;
		this.neighbors = new Neighbor[this.k];//a list of the k nearest neighbors
		this.freq = 1;//keep every instance in features		
		
		this.skip = new HashSet<Integer>();
	}

	@Override
	public void train(Matrix features, Matrix labels) throws Exception {		
		this.features = features;
		this.labels = labels;		
	}

	@Override
	public void predict(double[] features, double[] labels) throws Exception {//this gets an instance of our test set
		if (this.regression) {
			this.regression(features, labels);
		}
		else {
			this.nearest(features, labels);
		}
	}
	
	//counts all instances in this column that are equal to val, but also have a target value of <target>
	private int countVal(int col, double value, double target) {
		int count = 0;
		for (int i = 0; i < this.features.rows(); i++) {
			if (this.features.get(i,col) == value && this.labels.get(i, 0) == target) {
				count++;
			}
		}
		return count;
	}
	
	//calculates the distance from this instance to the instance at index <instance>
	//this function is used for nominal features
	private double vdm(double[] features, double[] labels, int instance, int col) {
		int numberOfClasses = this.labels.valueCount(0);
		if (NearestNeighbor.DISTANCE_DEBUG) {
			System.out.println("Number of output classes: " + numberOfClasses);
		}
		double sum = 0;
		
		for (int i = 0; i < numberOfClasses; i++) {//for each output class
			double Nax = this.features.countVal(col, features[col]);//the number of instances in training set that have value x for col		
			double Naxc = this.countVal(col, features[col], i);//get all instances of this feature that have value x for col and the label has value
			
			double Nay = this.features.countVal(col, this.features.get(instance, col));
			double Nayc = this.countVal(col, this.features.get(instance, col), i);
			
			if (NearestNeighbor.DISTANCE_DEBUG) {
				System.out.println("Calculating for class: " + i);
			}
			
			sum += Math.pow(Math.abs((Naxc/Nax) - (Nayc/Nay)),2);
		}		
		
		return sum;
	}
	
	//this function returns the distance between this instance and the instance at <instance>
	//this function is used for linear features
	private double normalizedDiff(double[] features, double[] labels, int instance, int col) throws Exception {
		double SD = this.features.columnSD(col);
		return Math.abs(features[col] - this.features.get(instance, col)) / (4*SD);
	}
	
	//this function calculates the distance between the instance represented by features, and labels to the instance contain within this.features[instance]
	public double calcDistance(double[] features, double[] labels, int instance) throws Exception {
		double distance = 0;
		if (!this.hetogeneous) {//if the dataset is homogeneous (only linear attributes)
			for (int col = 0; col < this.features.cols(); col++) {//calculate the distance from this instance
				if (this.skip.contains(col)) {
					continue;
				}
				distance += Math.abs(features[col] - this.features.get(instance, col));
			}
		}
		else {//the dataset has linear and nominal attributes
			for (int col = 0; col < this.features.cols(); col++) {//for each attribute
				//check if either feature is missing
				if (features[col] == Matrix.MISSING || this.features.get(instance, col) == Matrix.MISSING) {
					distance += 1;
				}
				else if (this.features.valueCount(col) == 0) {//check if feature is linear
					distance += Math.pow(this.normalizedDiff(features, labels, instance, col), 2);
				}
				else {//feature is nominal
					distance += Math.pow(this.vdm(features, labels, instance, col), 2);
				}
			}
			distance = Math.sqrt(distance);
		}
		
		return distance;
	}
	
	public void findNearestNeighbors(double[] features, double[] labels) throws Exception {
		//reset neighbors array
		for (int i = 0; i < this.k; i++) {
			this.neighbors[i] = new Neighbor();
		}
		
		for (int i = 0; i < this.features.rows(); i++) {//for every instance in the training set
			double distance = calcDistance(features, labels, i);
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
					System.out.println("Saved new nearest neighbor: " + farthestDistance + ", " + this.labels.get(i,0));
				}
				//save the new closest neighbor
				this.neighbors[farthestIndex].setDistance(distance);
				this.neighbors[farthestIndex].setLabel(this.labels.get(i,0));
			}
		}
	}
	
	public void regression(double[] features, double[] labels) throws Exception {		
		this.findNearestNeighbors(features, labels);//find the k nearest neighbors, they will be saved in this.neighbors
		
		if (NearestNeighbor.REGRESSION_DEBUG) {											
			System.out.print("[");
			for (int i = 0; i < this.k; i++) {
				System.out.print(this.neighbors[i].toString());
			}
			System.out.println("]");
		}
		
		double numerator = 0;
		double denominator = 0;
		
		if (!this.useDistanceWeighting) {
			denominator = this.k;
		}
		
		for (int i = 0; i < this.k; i++) {
			if (!this.useDistanceWeighting) {
				numerator += this.neighbors[i].getLabel();
			}
			else {
				numerator += ((1/this.neighbors[i].getDistance()) * this.neighbors[i].getLabel());
				denominator += (1/this.neighbors[i].getDistance());
			}
		}
		labels[0] = (numerator/denominator);
	}
	
	public void nearest(double[] features, double[] labels) throws Exception {
		int numberOfOutputClasses = this.labels.countUnique();
		this.findNearestNeighbors(features, labels);//find the k nearest neighbors, they will be saved in this.neighbors
		
		if (NearestNeighbor.DEBUG_VOTING) {
			System.out.print("Neighbors: [");
			for (int i = 0; i < this.k; i++) {
				System.out.print(this.neighbors[i].toString());
			}
			System.out.println("]");
		}		
		
		double outputClasses[] = new double[numberOfOutputClasses];//an array to hold all of the votes for each class
		
		//find the class that has the most votes
		if (!this.useDistanceWeighting) {
			for (int i = 0; i < this.k; i++) {
				outputClasses[(int) this.neighbors[i].getLabel()] += 1;//add a vote
			}
		}
		else {
			double denominator = 0, numerator = 0;
			for (int j = 0; j < this.k; j++) {//for each neighbor
				denominator += (1.0 / (Math.pow(this.neighbors[j].getDistance(), 2) ));
			}
			
			
			for (int i = 0; i < numberOfOutputClasses; i++) {//for each output class
				numerator = 0;
				for (int j = 0; j < this.k; j++) {//for each neighbor
					if ((int) this.neighbors[j].getLabel() == i) {//if the neighbor's output class == the current class we are looking at
						numerator += (1.0/(Math.pow(this.neighbors[j].getDistance(),2)));
					}
				}
				
				outputClasses[i] += (numerator/denominator);
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
	
	public void personalExperiment(int attributes, Matrix testFeatures, Matrix testLabels) throws Exception {		
		System.out.println("Starting experiment: ");
		//attributes = 5;
		for (int skipNum = 1; skipNum < attributes; skipNum++) {//for all values of columns to skip
			System.out.println("Skipping " + skipNum + " attributes");
			for (int start = 0; start < attributes; start++) {				//for all columns
				for (int index = start, i = 0; i < skipNum; index++, i++) {	//increment the index to skip				
					if (index >= attributes) {
						index = 0;
					}				
					this.skip.add(index);//add the index to skip					
				}					
				
				System.out.print("skipping: " + this.skip);
				//double accuracy = this.measureAccuracy(testFeatures, testLabels, null);
				//System.out.println(", Accuracy: " + accuracy);
				System.out.println();
				this.skip.clear();
			}	
			System.out.println();
		}
				
	}
}
