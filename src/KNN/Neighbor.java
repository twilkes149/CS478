package KNN;

public class Neighbor {
	private double distance;
	private double label;
	
	public Neighbor() {
		this.distance = Double.MAX_VALUE;
		this.label = Integer.MAX_VALUE;
	}
	
	public double getDistance() {
		return distance;
	}
	public void setDistance(double distance) {
		this.distance = distance;
	}
	public double getLabel() {
		return label;
	}
	public void setLabel(double label) {
		this.label = label;
	}
	
	public String toString() {
		String result = "{" + distance + ", " + label + "}";
		return result;
	}
}
