package Clustering;
import labs.Matrix;

public class Cluster {
	private Matrix data;
	private double[] centroid;
	
	public Cluster(int numCols) {
		this.centroid = null;
		this.data = new Matrix();
		this.data.setSize(0, numCols);
	}
	
	public Cluster(double[] instance, int numCols) {//creates a cluster
		this.centroid = instance;
		data = new Matrix();//create an empty matrix
		this.data.setSize(0, numCols);
	}
	
	public void setHeaders(Matrix data_) {
		this.data.setM_attr_name(data_.getM_attr_name());
		this.data.setM_enum_to_str(data_.getM_enum_to_str());
		this.data.setM_str_to_enum(data_.getM_str_to_enum());
	}
	
	public Matrix getData() {
		return data;
	}

	public void setData(Matrix data) {
		this.data = data;
	}
	
	public double[] getCentroid() {
		return this.centroid;
	}
	
	public void setCentroid(double[] c) {
		this.centroid = c;
	}
	
	//returns how many data instances this cluster contains
	public int numInstances() {
		return this.data.rows();
	}
	
	//function to get a particular instance
	public double[] get(int index) {
		return this.data.row(index);
	}
	
	//removes a data instance from this cluster
	public double[] remove(int index) {
		return this.data.remove(index);
	}
	
	//adds an instance to this cluster
	public void add(double[] instance) throws Exception {
		this.data.add(instance);
	}
	
	//recalculates the centroid point based on the instances contained within this cluster
	public void calcNewCentroid() {
		this.centroid = new double[this.centroid.length];
		for (int i = Clustering.START; i < this.centroid.length; i++) {
			if (this.data.valueCount(i) == 0) {//continuous data
				this.centroid[i] = this.data.columnMean(i);
			}
			else {//nominal data
				this.centroid[i] = this.data.mostCommonValue(i);
			}
		}
	}
	
	
	public double getSSE() throws Exception {
		double sum = 0;
		for (int i = 0; i < this.data.rows(); i++) {//for each row
			sum += Math.pow(this.data.getDistance(i, this.centroid), 2);//calc the distance between the instance and the centroid			
		}			
		return sum;
	}
	
	//This function returns the average distance of the instance at <index> to each other point in this cluster
	private double getAverageSeparation(int index) throws Exception {
		double sum = 0;		
				
		for (int j = 0; j < this.numInstances(); j++) {//for every other instance in the dataset							
			if (index == j) {
				continue;
			}
			sum += this.data.getDistance(this.data.row(index), this.data.row(j));//get distance between two instances
		}
		
		return sum/this.numInstances();
	}
	
	//calculates the average distance from an instance at <index> to every point in the given cluster
	private double getAverageDistance(int index, Cluster c) throws Exception {
		double sum = 0;
		
		for (int i = 0; i < c.numInstances(); i++) {//for all instances in the given cluster
			sum += this.data.getDistance(this.data.row(index), c.get(i));//calc distance betwen curr point and point in given cluster
		}
		
		return sum/c.numInstances();
	}
	
	//calculates the silhouette for the cluster compared against the given cluster
	public double calcSilhouette(Cluster c) throws Exception {
		double sum = 0;
		
		for (int i = 0; i < this.numInstances(); i++) {//for each instance
			//get a silhouette score for each instance and total it
			double a = this.getAverageSeparation(i);
			double b = this.getAverageDistance(i, c);
			
			sum += ((b-a) / Math.max(a, b));			
		}
		
		return sum/this.numInstances();//return the average silhouette score for this cluster
	}
}
