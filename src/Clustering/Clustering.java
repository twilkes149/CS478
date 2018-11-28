package Clustering;

import java.util.Random;

import backpropogation.Network;
import labs.Matrix;
import labs.SupervisedLearner;

public class Clustering {
	private int k;//used to define a number of clusters
	private boolean useInit;//used to determine if init centroids are random or the first k entries
	private int maxEpoch;
	private Cluster[] clusters;
	
	private Matrix data;//used to hold	
	
	public static final boolean STATS_DEBUG = true;
	public static final boolean INSTANCE_DEBUG = true;
	
	public static final int START = 0;//used when having a col id
	
	public Clustering(int k_, Matrix data_) throws Exception {
		this.k = k_;
		this.useInit = true;
		this.maxEpoch = 200;//max number of iterations to refine the clusters
		this.clusters = new Cluster[this.k];//generate k empty clusters
		
		for (int i = 0; i < this.k; i++) {
			this.clusters[i] = new Cluster(data_.cols());
			this.clusters[i].setHeaders(data_);
		}
		
		this.generateClusters(data_);
	}

	//meant to be run once on init to generate the initial set of clusters	
	private void generateClusters(Matrix data_) throws Exception {
		String test = "";
		this.data = data_;
		Random rand = new Random();
		
		for (int i = 0; i < this.k; i++) {//grab k random data instances to be the initial centroids
			
			int index=i;			
			if (!this.useInit) {
				index = rand.nextInt(this.data.rows());
			}
			this.clusters[i].setCentroid(this.data.row(index));//remove the instance from the data and set it as the cluster's centroid
		}
		
		for (int i = 0; i < this.data.rows(); i++) {//for each data instance
			int closestClusterIndex = 0;
			double closestDistance = Double.MAX_VALUE;
			
			for (int j = 0; j < this.k; j++) {//for each cluster
				double distance = this.data.getDistance(i, this.clusters[j].getCentroid()); //calc distance to cluster's centroid
				
				if (distance < closestDistance) {//if the instance is closer to this cluster than any previous clusters
					closestDistance = distance;
					closestClusterIndex = j;
				}
			}
			this.clusters[closestClusterIndex].add(this.data.row(i));//assign data instance to closest cluster
			test += (i) + "=" + closestClusterIndex + " ";
			if ((i) % 9 == 0) {
				test += "\n";
			}
		}
		if (Clustering.INSTANCE_DEBUG) {
			System.out.println(test);
		}
		
	}
	
	//makes one pass over the data to refine the cluster
	//returns true if the clusters have converged (no data has changed clusters)
	private boolean refineClusters() throws Exception {
		boolean moved = false;
		String test = "";
		int count = 0;
		for (int i = 0; i < this.k; i++) {//for each cluster
			//for each data instance in cluster
				//for each cluster
					//calc distance of data instance to cluster centroid
				//move data instance to closest cluster			
			for (int j = 0; j < this.clusters[i].numInstances(); count++) {//for each data instance in the cluster
				
				int closestClusterIndex = i;
				double closestDistance = Double.MAX_VALUE;
				
				//calculate which cluster, if any the data instance should be moved to
				for (int clusterIndex = 0; clusterIndex < this.k; clusterIndex++) {//for each cluster
					double[] clusterCentroid = this.clusters[clusterIndex].getCentroid();
					
					double distance = this.data.getDistance(clusterCentroid, this.clusters[i].get(j));
					
					if (distance < closestDistance) {//if the instance is closer to this cluster than any previous clusters
						closestDistance = distance;
						closestClusterIndex = clusterIndex;
					}
				}
				
				if (closestClusterIndex != i) {//if the cluster we are moving the data to is not the cluster it already is in
					moved = true;					
					test += (int) this.clusters[i].get(j)[0] + ".m=" + closestClusterIndex + " ";
					this.clusters[closestClusterIndex].add(this.clusters[i].remove(j));//pop the instance out of the curr cluster, and add it to the closest cluster					
				}
				else {
					test += (int) this.clusters[i].get(j)[0] + ".s=" + i + " ";
					j++;//if we don't move the data, move on to the next instance					
				}
			}
			test +="\n";
		}		
		if (Clustering.INSTANCE_DEBUG) {
			System.out.println(test);
			System.out.println("Refined: " + count + "instances");
		}
		return !moved;
	}
	
	public void train() throws Exception {	
		//this.data.print();
		if (Clustering.STATS_DEBUG) {
			System.out.println("*******************");
			System.out.println("Iteration 1");
			System.out.println("*******************");
			System.out.println(this.toString() + "\n");
		}
		for (int j = 0; j < this.k; j++) {//for each cluster
			this.clusters[j].calcNewCentroid();//re-calculate the centroid
		}
		
		
		for (int i = 0; i < this.maxEpoch; i++) {//for the max number of iterations			
			if (Clustering.STATS_DEBUG) {
				System.out.println("*******************");
				System.out.println("Iteration " + (i+2));
				System.out.println("*******************");
				System.out.println(this.toString() + "\n");
			}
			
			if (this.refineClusters()) {//if the clusters have converged, stop refining them
				System.out.println("Clusters have converged");
				return;
			}
			else {
				for (int j = 0; j < this.k; j++) {//for each cluster
					this.clusters[j].calcNewCentroid();//re-calculate the centroid
				}
			}
		}
	}
	
	public String toString(){
		String result = "Clusters:\n";
		double sum = 0;
		for (int i = 0; i < this.k; i++) {
			result += "\n  Cluster ["+ i +"]: Size: " + this.clusters[i].numInstances() + " Centroid " + ": " + this.data.rowToString(this.clusters[i].getCentroid());
			try {
				sum += this.clusters[i].getSSE();
			} catch (Exception e) {				
				e.printStackTrace();
			}
		}
		result += "\nSSE: " + sum;
		return result;
		
	}

}
