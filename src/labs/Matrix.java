// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------
package labs;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Scanner;
import java.util.Set;
import java.util.TreeMap;

import backpropogation.Network;

import java.util.Map.Entry;
import java.util.Random;
import java.util.Iterator;
import java.util.Map;
import java.io.File;
import java.io.FileNotFoundException;
import java.lang.Exception;

public class Matrix {
	// Data
	ArrayList< double[] > m_data;

	// Meta-data
	ArrayList< String > m_attr_name;
	ArrayList< TreeMap<String, Integer> > m_str_to_enum;
	ArrayList< TreeMap<Integer, String> > m_enum_to_str;

	public static double MISSING = Double.MAX_VALUE; // representation of missing values in the dataset

	// Creates a 0x0 matrix. You should call loadARFF or setSize next.
	public Matrix() {}

	// Copies the specified portion of that matrix into this matrix
	public Matrix(Matrix that, int rowStart, int colStart, int rowCount, int colCount) {
		m_data = new ArrayList< double[] >();
		for(int j = 0; j < rowCount; j++) {
			double[] rowSrc = that.row(rowStart + j);
			double[] rowDest = new double[colCount];
			for(int i = 0; i < colCount; i++)
				rowDest[i] = rowSrc[colStart + i];
			m_data.add(rowDest);
		}
		m_attr_name = new ArrayList<String>();
		m_str_to_enum = new ArrayList< TreeMap<String, Integer> >();
		m_enum_to_str = new ArrayList< TreeMap<Integer, String> >();
		for(int i = 0; i < colCount; i++) {
			m_attr_name.add(that.attrName(colStart + i));
			m_str_to_enum.add(that.m_str_to_enum.get(colStart + i));
			m_enum_to_str.add(that.m_enum_to_str.get(colStart + i));
		}
	}
	
	public int getNumberOfPossibleAttributeValues(int column) {
		return 0;
	}
	
	public ArrayList<TreeMap<String, Integer>> getM_str_to_enum() {
		return m_str_to_enum;
	}

	public void setM_str_to_enum(ArrayList<TreeMap<String, Integer>> m_str_to_enum) {
		this.m_str_to_enum = m_str_to_enum;
	}

	public ArrayList<TreeMap<Integer, String>> getM_enum_to_str() {
		return m_enum_to_str;
	}

	public void setM_enum_to_str(ArrayList<TreeMap<Integer, String>> m_enum_to_str) {
		this.m_enum_to_str = m_enum_to_str;
	}
	
	public ArrayList<String> getM_attr_name() {
		return m_attr_name;
	}

	public void setM_attr_name(ArrayList<String> m_attr_name) {
		this.m_attr_name = m_attr_name;
	}

	public String getAttributeName(int index) {
		//System.out.print(m_enum_to_str.get(j).get((int)r[j]));
		
		return this.m_attr_name.get(index);
	}

	//this function returns an array where the lenght is the number of unique elements inside that, and each element is how many of that value there is
	public double[] countUnique(Matrix that, int col) throws Exception {		
		HashSet<Double> values = (HashSet<Double>) that.getFeatureValues(that, col);//gets all values for this column
		int uniqueValues = values.size();
		double[] valuesCount = new double[uniqueValues];
				
		int index = 0;
		for (Iterator<Double> i = values.iterator(); i.hasNext() && index < uniqueValues; index++) {
			double temp = i.next();			
			valuesCount[index] = that.countVal(col, temp);			
		}
				
		//System.out.println("number of each class: " + Network.arrayToString(valuesCount));
		return valuesCount;
	}
	
	
	//the first element is the selected features, the second element is the labels to match
	public Matrix[] select(Matrix that, int col, double value, Matrix labels, int numValues) {		
		Matrix m[] = {new Matrix(), new Matrix()};
		
		
		//int numValues =that.countVal(col, value); 
		m[0].setSize(numValues, that.cols());
		m[0].setM_enum_to_str(that.getM_enum_to_str());
		m[0].setM_str_to_enum(that.getM_str_to_enum());
		m[0].setM_attr_name(that.getM_attr_name());
		
		m[1].setSize(numValues, 1);
		m[1].setM_enum_to_str(labels.getM_enum_to_str());
		m[1].setM_str_to_enum(labels.getM_str_to_enum());
		m[1].setM_attr_name(labels.getM_attr_name());
				
		
		for (int i = 0, index=0; i < that.rows(); i++) {
			if (that.get(i, col) == value) {//getting the row and putting it into the new row
				for (int j = 0; j < m[0].cols(); j++) {
					m[0].set(index, j, that.get(i, j));//select the data rows					
				}
				m[1].set(index, 0, labels.get(i, 0));//select the labels
				index++;
			}
		}		
		return m;
	}
	
	//returns how many the number of instances were col = value
	public int countVal(int col, double value) {
		int count = 0;
		for (int i = 0; i < this.rows(); i++) {
			if (this.get(i,col) == value) {
				count++;
			}
		}
		return count;
	}
	
	//counts how many unique elements are in this array
	public int countUnique() throws Exception {
		if (this.cols() != 1) {//we only want to deal with 1 dimensional matrices
			throw new Exception("Expected matrix to have one column");
		}
		Set<Integer> attributes = new HashSet<Integer>();
		for (int i = 0; i < this.rows(); i++) {
			attributes.add((int) this.get(i, 0));
		}
		return attributes.size();
	}
	
	//returns a list of all values of a given column
	public Set<Double> getFeatureValues(Matrix set, int colIndex) throws Exception {
		HashSet<Double> values = new HashSet<Double>();
		for (int i = 0; i < set.rows(); i++) {
			values.add(set.get(i, colIndex));
		}
		return values;
	}

	// Adds a copy of the specified portion of that matrix to this matrix
	public void add(Matrix that, int rowStart, int colStart, int rowCount) throws Exception {
		if(colStart + cols() > that.cols())
			throw new Exception("out of range");
		for(int i = 0; i < cols(); i++) {
			if(that.valueCount(colStart + i) != valueCount(i))
				throw new Exception("incompatible relations");
		}
		for(int j = 0; j < rowCount; j++) {
			double[] rowSrc = that.row(rowStart + j);
			double[] rowDest = new double[cols()];
			for(int i = 0; i < cols(); i++)
				rowDest[i] = rowSrc[colStart + i];
			m_data.add(rowDest);
		}
	}

	// Resizes this matrix (and sets all attributes to be continuous)
	public void setSize(int rows, int cols) {
		m_data = new ArrayList< double[] >();
		for(int j = 0; j < rows; j++) {
			double[] row = new double[cols];
			m_data.add(row);
		}
		m_attr_name = new ArrayList<String>();
		m_str_to_enum = new ArrayList< TreeMap<String, Integer> >();
		m_enum_to_str = new ArrayList< TreeMap<Integer, String> >();
		for(int i = 0; i < cols; i++) {
			m_attr_name.add("");
			m_str_to_enum.add(new TreeMap<String, Integer>());
			m_enum_to_str.add(new TreeMap<Integer, String>());
		}
	}

	// Loads from an ARFF file
	public void loadArff(String filename) throws Exception, FileNotFoundException {
		m_data = new ArrayList<double[]>();
		m_attr_name = new ArrayList<String>();
		m_str_to_enum = new ArrayList< TreeMap<String, Integer> >();
		m_enum_to_str = new ArrayList< TreeMap<Integer, String> >();
		boolean READDATA = false;
		Scanner s = new Scanner(new File(filename));
		while (s.hasNext()) {
			String line = s.nextLine().trim();
			if (line.length() > 0 && line.charAt(0) != '%') {
				if (!READDATA) {
					
					Scanner t = new Scanner(line);
					String firstToken = t.next().toUpperCase();
					
					if (firstToken.equals("@RELATION")) {
						String datasetName = t.nextLine();
					}
					
					if (firstToken.equals("@ATTRIBUTE")) {
						TreeMap<String, Integer> ste = new TreeMap<String, Integer>();
						m_str_to_enum.add(ste);
						TreeMap<Integer, String> ets = new TreeMap<Integer, String>();
						m_enum_to_str.add(ets);

						Scanner u = new Scanner(line);
						if (line.indexOf("'") != -1) u.useDelimiter("'");
						u.next();
						String attributeName = u.next();
						if (line.indexOf("'") != -1) attributeName = "'" + attributeName + "'";
						m_attr_name.add(attributeName);

						int vals = 0;
						String type = u.next().trim().toUpperCase();
						if (type.equals("REAL") || type.equals("CONTINUOUS") || type.equals("INTEGER")) {
						}
						else {
							try {
								String values = line.substring(line.indexOf("{")+1,line.indexOf("}"));
								Scanner v = new Scanner(values);
								v.useDelimiter(",");
								while (v.hasNext()) {
									String value = v.next().trim();
									if(value.length() > 0)
									{
										ste.put(value, new Integer(vals));
										ets.put(new Integer(vals), value);
										vals++;
									}
								}
							}
							catch (Exception e) {
								throw new Exception("Error parsing line: " + line + "\n" + e.toString());
							}
						}
					}
					if (firstToken.equals("@DATA")) {
						READDATA = true;
					}
				}
				else {
					double[] newrow = new double[cols()];
					int curPos = 0;

					try {
						Scanner t = new Scanner(line);
						t.useDelimiter(",");
						while (t.hasNext()) {
							String textValue = t.next().trim();
							//System.out.println(textValue);

							if (textValue.length() > 0) {
								double doubleValue;
								int vals = m_enum_to_str.get(curPos).size();
								
								//Missing instances appear in the dataset as a double defined as MISSING
								if (textValue.equals("?")) {
									doubleValue = MISSING;
								}
								// Continuous values appear in the instance vector as they are
								else if (vals == 0) {
									doubleValue = Double.parseDouble(textValue);
								}
								// Discrete values appear as an index to the "name" 
								// of that value in the "attributeValue" structure
								else {
									doubleValue = m_str_to_enum.get(curPos).get(textValue);
									if (doubleValue == -1) {
										throw new Exception("Error parsing the value '" + textValue + "' on line: " + line);
									}
								}
								
								newrow[curPos] = doubleValue;
								curPos++;
							}
						}
					}
					catch(Exception e) {
						throw new Exception("Error parsing line: " + line + "\n" + e.toString());
					}
					m_data.add(newrow);
				}
			}
		}
	}

	// Returns the number of rows in the matrix
	public int rows() { return m_data.size(); }

	// Returns the number of columns (or attributes) in the matrix
	public int cols() { return m_attr_name.size(); }

	// Returns the specified row
	public double[] row(int r) { return m_data.get(r); }

	// Returns the element at the specified row and column
	public double get(int r, int c) { return m_data.get(r)[c]; }

	// Sets the value at the specified row and column
	public void set(int r, int c, double v) { row(r)[c] = v; }

	// Returns the name of the specified attribute
	public String attrName(int col) { return m_attr_name.get(col); }

	// Set the name of the specified attribute
	public void setAttrName(int col, String name) { m_attr_name.set(col, name); }

	// Returns the name of the specified value
	public String attrValue(int attr, int val) { return m_enum_to_str.get(attr).get(val); }

	// Returns the number of values associated with the specified attribute (or column)
	// 0=continuous, 2=binary, 3=trinary, etc.
	public int valueCount(int col) { return m_enum_to_str.get(col).size(); }

	// Shuffles the row order
	public void shuffle(Random rand) {
		for(int n = rows(); n > 0; n--) {
			int i = rand.nextInt(n);
			double[] tmp = row(n - 1);
			m_data.set(n - 1, row(i));
			m_data.set(i, tmp);
		}
	}

	// Shuffles the row order with a buddy matrix 
	public void shuffle(Random rand, Matrix buddy) {
		for (int n = rows(); n > 0; n--) {
			int i = rand.nextInt(n);
			double[] tmp = row(n - 1);
			m_data.set(n - 1, row(i));
			m_data.set(i, tmp);


			double[] tmp1 = buddy.row(n - 1);
			buddy.m_data.set(n - 1, buddy.row(i));
			buddy.m_data.set(i, tmp1);
		}
	}

	// Returns the mean of the specified column
	public double columnMean(int col) {
		double sum = 0;
		int count = 0;
		for(int i = 0; i < rows(); i++) {
			double v = get(i, col);
			if(v != MISSING)
			{
				sum += v;
				count++;
			}
		}
		return sum / count;
	}

	// Returns the min value in the specified column
	public double columnMin(int col) {
		double m = MISSING;
		for(int i = 0; i < rows(); i++) {
			double v = get(i, col);
			if(v != MISSING)
			{
				if(m == MISSING || v < m)
					m = v;
			}
		}
		return m;
	}

	// Returns the max value in the specified column
	public double columnMax(int col) {
		double m = MISSING;
		for(int i = 0; i < rows(); i++) {
			double v = get(i, col);
			if(v != MISSING)
			{
				if(m == MISSING || v > m)
					m = v;
			}
		}
		return m;
	}
	
	//returns the standard deviation of a column
	public double columnSD(int col) throws Exception{
		if (this.valueCount(col) != 0) {
			throw new Exception("Column must be continuous");
		}
		else {
			double N = this.rows();
			double mean = this.columnMean(col);
			
			double sum = 0;
			for (int i = 0; i < this.rows(); i++) {
				sum += Math.pow((this.get(i, col) - mean), 2);
			}
			
			return Math.sqrt(sum/N);
		}
	}

	// Returns the most common value in the specified column
	public double mostCommonValue(int col) {
		TreeMap<Double, Integer> tm = new TreeMap<Double, Integer>();
		for(int i = 0; i < rows(); i++) {
			double v = get(i, col);
			if(v != MISSING)
			{
				Integer count = tm.get(v);
				if(count == null)
					tm.put(v, new Integer(1));
				else
					tm.put(v, new Integer(count.intValue() + 1));
			}
		}
		int maxCount = 0;
		double val = MISSING;
		Iterator< Entry<Double, Integer> > it = tm.entrySet().iterator();
		while(it.hasNext())
		{
			Entry<Double, Integer> e = it.next();
			if(e.getValue() > maxCount)
			{
				maxCount = e.getValue();
				val = e.getKey();
			}
		}
		return val;
	}

	public void normalize() {
		for(int i = 0; i < cols(); i++) {
			if(valueCount(i) == 0) {
				double min = columnMin(i);
				double max = columnMax(i);
				for(int j = 0; j < rows(); j++) {
					double v = get(j, i);
					if(v != MISSING)
						set(j, i, (v - min) / (max - min));
				}
			}
		}
	}

	public void print() {
		System.out.println("@RELATION Untitled");
		for(int i = 0; i < m_attr_name.size(); i++) {
			System.out.print("@ATTRIBUTE " + m_attr_name.get(i));
			int vals = valueCount(i);
			if(vals == 0)
				System.out.println(" CONTINUOUS");
			else
			{
				System.out.print(" {");
				for(int j = 0; j < vals; j++) {
					if(j > 0)
						System.out.print(", ");
					System.out.print(m_enum_to_str.get(i).get(j));
				}
				System.out.println("}");
			}
		}
		System.out.println("@DATA");
		for(int i = 0; i < rows(); i++) {
			double[] r = row(i);
			for(int j = 0; j < r.length; j++) {
				if(j > 0)
					System.out.print(", ");
				if(valueCount(j) == 0)
					System.out.print(r[j]);
				else
					System.out.print(m_enum_to_str.get(j).get((int)r[j]));
			}
			System.out.println("");
		}
	}
}
