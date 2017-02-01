import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Stack;

/**
 * Fill in the implementation details of the class DecisionTree using this file. Any methods or
 * secondary classes that you want are fine but we will only interact with those methods in the
 * DecisionTree framework.
 * 
 * You must add code for the 1 member and 4 methods specified below.
 * 
 * See DecisionTree for a description of default methods.
 */
public class DecisionTreeImpl extends DecisionTree {
	private DecTreeNodeImpl root;
	//ordered list of class labels
	private List<String> labels; 
	//ordered list of attributes
	private List<String> attributes; 
	//map to ordered discrete values taken by attributes
	private Map<String, List<String>> attributeValues;
	//
	private List<HashMap<String, List<Integer>>> neededValues;

	/**
	 * Answers static questions about decision trees.
	 */
	DecisionTreeImpl() {
		// no code necessary this is void purposefully
	}

	/**
	 * Build a decision tree given only a training set.
	 * 
	 * @param train: the training set
	 */
	DecisionTreeImpl(DataSet train) {

		this.labels = train.labels;
		this.attributes = train.attributes;
		this.attributeValues = train.attributeValues;
		// TODO: add code here
		List<HashMap<String, List<Integer>>> neededValues = new ArrayList<HashMap<String, List<Integer>>>();
		this.root = DecisionTreeImplHelper(train.instances, train.attributes, null);
	}

	@Override
	public String classify(Instance instance) 
	{
		// TODO: add code here
		if (this.root == null)
		{
			return null;
		}
		//start from the root and move downward
		DecTreeNode currentNode = this.root;
		//once we hit a terminal node, then we break out of this, and we return that terminal nodes value
		while (!currentNode.terminal)
		{
			String currentNodeAttr = currentNode.attribute; //like A1
			int attrIndex = getAttributeIndex(currentNodeAttr); 
			currentNode = currentNode.children.get(getAttributeValueIndex(currentNode.attribute, instance.attributes.get(attrIndex)));
		}
		return currentNode.label;
	}

	@Override
	public void rootInfoGain(DataSet train) 
	{
		this.labels = train.labels;
		this.attributes = train.attributes;
		this.attributeValues = train.attributeValues;
		// TODO: add code here
		// makes a list of maps (each corresponding to an attribute A1, A2, etc
		// and then for each one of them, go through all the (global) possible 
		// values and zero their respective counts. I.e. A1-> x = 0
		// list of attribute maps each attribute A1, A2, ... A(N) has a map
		List<HashMap<String, List<Integer>>> listOfAttributeHashMaps = new ArrayList<HashMap<String, List<Integer>>>();
		// zero out the map's count at every possible attribute and each attribute's values
		for (String indvAttribute : this.attributes)
		{
			HashMap<String, List<Integer>> map = new HashMap<String, List<Integer>>();
			// going through each value and setting its count to 0
			for (String value : this.attributeValues.get(indvAttribute))
			{
				//first index is the total count, while each subsequent index corresponds
				//to the individual label counts
				List<Integer> counts = new ArrayList<Integer>();
				counts.add(0, 0);
				for (int i = 1; i <= this.labels.size(); i++)
				{
					counts.add(i, 0);
				}
				map.put(value, counts);
			}
			listOfAttributeHashMaps.add(map);
		}
		// If-else counts the total number of G and B's that occur. 
		// For loop after retrieves the List<Integer> counts which is specific 
		// to a single A1 -> value (attribute->value). Index 0 of this list 
		// is the total count and preceeding indexes correspond to the label 
		// classes (G and B).
		//
		// count each occurrence of G and B using this map
		HashMap<String, Integer> hashMapOfLabels = new HashMap<String, Integer>();
		// loop through each instance
		for (Instance indvInstance : train.instances)
		{
			// iterate (values) and count for each label/class identified
			// within each of the instances
			if (hashMapOfLabels.get(indvInstance.label) == null)
			{
				hashMapOfLabels.put(indvInstance.label, 1);
			}
			else
			{
				hashMapOfLabels.put(indvInstance.label, hashMapOfLabels.get(indvInstance.label) + 1);
			}

			// increment counts for the various values in the instance 
			for (int i = 0; i < indvInstance.attributes.size(); i++)
			{
				// get the encoded list of values
				List<Integer> counts = listOfAttributeHashMaps.get(i).get(indvInstance.attributes.get(i));
				// always increment the total count
				int replacementTotal = counts.get(0) + 1;
				counts.remove(0);
				counts.add(0, replacementTotal);
				// get index of the label encountered
				int labelIndex = getLabelIndex(indvInstance.label) + 1;
				int newLabelCount = counts.get(labelIndex) + 1;
				counts.remove(labelIndex);
				counts.add(labelIndex, newLabelCount);
				//put the augmented count list back into the map
				listOfAttributeHashMaps.get(i).put(indvInstance.attributes.get(i), counts);
			}
			// transfers the needed attribute values and counts to the 
			// decision tree object for later use.
			for(HashMap<String, List<Integer>> hm : listOfAttributeHashMaps)
			{
				neededValues.add(hm);
			}
		}
		// Develops the entropy value H. Loops through G and B and performs a 
		// calculation that is added to the running total
		//
		//loop through key's (labels (G, B)) and start forming the H value through its formula
		double HValue = 0;
		double total = (double) train.instances.size();
		Iterator it = hashMapOfLabels.entrySet().iterator();
		while (it.hasNext()) 
		{
			Map.Entry pair = (Map.Entry)it.next();
			HValue += -((double) (int)pair.getValue()/total)
					*(Math.log((double) (int)pair.getValue()/total)/Math.log(2));
		}

		//
		// This loop passes going through A1, A2, A3,..., A(N) once
		// (only the passes in attributes still left to select from)
		//
		List<Double> condHList = new ArrayList<Double>(); 
		for (String indvAttribute : this.attributes){
			double condHValue = 0;
			//find conditional H (i.e. H(class | A1, A2, ...))
			int index = getAttributeIndex(indvAttribute);
			//only look at attributes left in our list
			HashMap<String, List<Integer>> map = listOfAttributeHashMaps.get(index);
			it = map.entrySet().iterator();
			//look at each attr value
			while (it.hasNext()) 
			{
				//attr value and counts pair
				Map.Entry<String, List<Integer>> pair = (Map.Entry<String, List<Integer>>)it.next();
				List<Integer> counts = pair.getValue();
				for(int i = 1; i < counts.size(); i++)
				{
					// --------------------------------------------------------
					// H(Y | X) -> H(Class Label | Attribute) =
					// --------------------------------------------------------
					// (Count of Attribute 1 / total) * H(CoA1<G>/CoA1, CoA1<B>/CoA1) 
					// + (Count of Attribute 2 / total) * H(CoA2<G>/CoA2, CoA2<B>/CoA2)
					// + (Count of Attribute 3 / total) * H(CoA3<G>/CoA3, CoA3<B>/CoA3)
					// ........
					// + (Count of Attribute N/ total) * H(CoAN<G>/CoAN, CoAN<B>/CoAN)
					// --------------------------------------------------------
					if (counts.get(i) != 0)
					{
						condHValue += -((double) counts.get(i)/counts.get(0))*((double) counts.get(0)/total)
								*(Math.log((double) counts.get(i)/counts.get(0))/Math.log(2));
					}
				}
			}
			// Add new Conditional Entropy value to end of list.
			condHList.add(condHValue);
		}
		// loop through conditional entropy value list
		// looking for the attribute providing largest
		// H(label) - H(label|attribute)
		double minCondH = condHList.get(0);
		int indexOfBestAttr = 0;
		for (int i = 0; i < condHList.size(); i++)
		{
			if(minCondH > condHList.get(i)){
				minCondH = condHList.get(i);
				indexOfBestAttr = i;
			}
		}
		//best attr will have same index in attributes that its condHValue had in the list of condHValues
		//return attributes.get(indexOfBestAttr);
		for(int i = 0; i < condHList.size(); i++){
			double hValTotal = HValue - condHList.get(i);
			System.out.format(train.attributes.get(i) + " %.5f\n", hValTotal);
		}

	}

	@Override
	public void printAccuracy(DataSet test) 
	{
		// TODO: add code here
		String results[] = pipeResults(test);
		double correct = 0.0;
		double total = test.instances.size();
		for (int i = 0; i < test.instances.size(); i ++)
			if (test.instances.get(i).label.equals(classify(test.instances.get(i))))
			{
				correct ++;
			}
		System.out.format( " %.5f\n",  correct / total);
	}
	
	/**
	 * Build a decision tree given a training set then prune it using a tuning 
	 * set.
	 * ONLY for extra credits
	 * @param train: the training set
	 * @param tune: the tuning set
	 */
	DecisionTreeImpl(DataSet train, DataSet tune) 
	{
		this(train);
		this.labels = train.labels;
		this.attributes = train.attributes;
		this.attributeValues = train.attributeValues;
		// TODO: add code here
		// only for extra credits
		
		//pruning loop
		double accuracy = determineAccuracy(tune);
		boolean done = false;
		while (!done) 
		{
			ArrayList<DecTreeNodeImpl> nodeListingFromRoot 
			= new ArrayList<DecTreeNodeImpl>();
			traverseTree(root, nodeListingFromRoot);
			int optimalIndex = -1;
			int i = 0;
			for (DecTreeNodeImpl node: nodeListingFromRoot) 
			{
				node.terminal = true;
				double tempAccuracy = determineAccuracy(tune);
				if (tempAccuracy > accuracy) {
					accuracy = tempAccuracy;
					optimalIndex = i;
				}
				node.terminal = false;
				i++;
			}
			if (optimalIndex != -1) 
			{
				nodeListingFromRoot.get(optimalIndex).children = null;
				nodeListingFromRoot.get(optimalIndex).terminal = true;
			} 
			else 
			{
				done = true;
			}	
		}	
	}

	@Override
	/**
	 * Print the decision tree in the specified format
	 */
	public void print() {

		printTreeNode(root, null, 0);
	}

	/**
	 * Prints the subtree of the node with each line prefixed by 4 * k spaces.
	 */
	public void printTreeNode(DecTreeNode p, DecTreeNode parent, int k) {
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < k; i++) {
			sb.append("    ");
		}
		String value;
		if (parent == null) {
			value = "ROOT";
		} else {
			int attributeValueIndex = this.getAttributeValueIndex(parent.attribute, p.parentAttributeValue);
			value = attributeValues.get(parent.attribute).get(attributeValueIndex);
		}
		sb.append(value);
		if (p.terminal) {
			sb.append(" (" + p.label + ")");
			System.out.println(sb.toString());
		} else {
			sb.append(" {" + p.attribute + "?}");
			System.out.println(sb.toString());
			for (DecTreeNode child : p.children) {
				printTreeNode(child, p, k + 1);
			}
		}
	}

	/**
	 * Helper function to get the index of the label in labels list
	 */
	private int getLabelIndex(String label) {
		for (int i = 0; i < this.labels.size(); i++) {
			if (label.equals(this.labels.get(i))) {
				return i;
			}
		}
		return -1;
	}

	/**
	 * Helper function to get the index of the attribute in attributes list
	 */
	private int getAttributeIndex(String attr) {
		for (int i = 0; i < this.attributes.size(); i++) {
			if (attr.equals(this.attributes.get(i))) {
				return i;
			}
		}
		return -1;
	}

	/**
	 * Helper function to get the index of the attributeValue in the list 
	 * for the attribute key in the attributeValues map
	 */
	private int getAttributeValueIndex(String attr, String value) {
		for (int i = 0; i < attributeValues.get(attr).size(); i++) 
		{
			if (value.equals(attributeValues.get(attr).get(i))) 
			{
				return i;
			}
		}
		return -1;
	}
	
	/*
	 * Auxiliary method used to actually build the Decision Tree.
	 */
	private DecTreeNodeImpl DecisionTreeImplHelper(List<Instance> instances, 
			List<String> attributes, List<Instance> parentInstances)
	{
		boolean allTheSame = true;
		String label;
		DecTreeNodeImpl node;
		// if list of examples is empty. 
		if (instances.isEmpty())
		{
			return majorityVotedClass(parentInstances);
		}
		// sets allTheSame to false if it finds any labels different
		label = instances.get(0).label;
		for (Instance i : instances)
		{
			if (!label.equals(i.label))
				allTheSame = false;
		}
		if (allTheSame)
		{
			// make new node with label class and return
			node = new DecTreeNodeImpl(label, null, null, true);
			return node;
		}
		// return most common from parent if no more attributes to split on
		if (attributes.isEmpty())
		{
			return majorityVotedClass(instances);
		}
		// find best attribute to split on TODO make helper fcn
		String bestAttr = determineBestAttribute(attributes, instances);
		// remove this attr from attributes list (not the master one though)
		List<String> updatedAttr = new ArrayList<String>(attributes);
		updatedAttr.remove(bestAttr);
		// make a new node N with this attribute
		node = new DecTreeNodeImpl(null, bestAttr, null, false);
		// for each possible value of the attribute:
		List<String> attrVals = this.attributeValues.get(bestAttr);
		for(String v: attrVals)
		{
			// make list of instances with that value
			List<Instance> valInstances = new ArrayList<Instance>();
			int attrIndex = getAttributeIndex(bestAttr);
			for(Instance i: instances)
			{
				if(i.attributes.get(attrIndex).equals(v))
					valInstances.add(i);
			}
			// new node = DecTreeNodeImplHelper(.....) (recursive call)
			DecTreeNodeImpl child = DecisionTreeImplHelper(valInstances, updatedAttr, instances);
			// make new node a child of node N
			child.setParentValue(v);
			node.children.add(child);
		}
		// return node
		return node;
	}
	
	/*
	 * Method used to determine the Tree's best attribute through 
	 * your typical information gain calculations.
	 */
	private String determineBestAttribute(List<String> attributes, List<Instance> instances)
	{
		// Creates a list of HashMap(s), each containing a list of maps
		// for every attribute and its values (ex: A1:...; A2:...; A3:..., etc.) 
		// each will have its very own map
		List<HashMap<String, List<Integer>>> listOfAttributeHashMaps = new ArrayList<HashMap<String, List<Integer>>>();
		// Sets each newly created HashMap label count to an initial count of
		// zero at every possible attribute and each attribute's values 
		for (String indvAttribute : this.attributes)
		{
			HashMap<String, List<Integer>> map = new HashMap<String, List<Integer>>();
			// Iterate through each value and sets count to 0
			for (String value : this.attributeValues.get(indvAttribute)){
				// The initial index is the totaled count, 
				// each of the following indices relates to the individual
				// label count.

				List<Integer> counts = new ArrayList<Integer>();
				counts.add(0, 0);
				for (int i = 1; i <= this.labels.size(); i++)
				{
					counts.add(i, 0);
				}
				map.put(value, counts);
			}
			listOfAttributeHashMaps.add(map);
		}
		// This section is responsible for counting
		// up each instance's G and B values with hashMapOfLabels
		HashMap<String, Integer> hashMapOfLabels = new HashMap<String, Integer>();
		// This loop will go through each instance's list of examples
		for (Instance indvInstance : instances)
		{
			//populate the count (value) for each label/class found in instances
			if (hashMapOfLabels.get(indvInstance.label) == null)
			{
				hashMapOfLabels.put(indvInstance.label, 1);
			}
			else
			{

				hashMapOfLabels.put(indvInstance.label, hashMapOfLabels.get(indvInstance.label) + 1);
			}
			// Poll totals for the various attribute values within
			// an instance 
			for (int i = 0; i < indvInstance.attributes.size(); i++)
			{
				// Retrieves a given list of integer values for each
				// example's list of attributes.
				List<Integer> counts = listOfAttributeHashMaps.get(i).get(indvInstance.attributes.get(i));
				// ------------------------------------------------------------
				// This portion assures the total is constantly updated
				// with the new total, and will remove the previous total's 
				// value in the list.
				// ------------------------------------------------------------
				int replacementTotal = counts.get(0) + 1;
				counts.remove(0);
				counts.add(0, replacementTotal);
				// ------------------------------------------------------------
				// Fetches specified index of a given label which we have
				// already accounted for. (+1, is there to account for 
				// the fixed 0 index, which is still keeping tabs on the 
				// total count and mapping it accordingly into the new HashMap)
				// ------------------------------------------------------------
				int labelIndex = getLabelIndex(indvInstance.label) + 1;
				// ------------------------------------------------------------
				// retrieves the old count from the counterList and adds to 
				// total of the count for the given label. 
				// ------------------------------------------------------------
				int newLabelCount = counts.get(labelIndex) + 1;
				counts.remove(labelIndex);
				counts.add(labelIndex, newLabelCount);
				// re-add/update the new values of the label counterList back
				// to the ArrayList of HashMaps
				listOfAttributeHashMaps.get(i).put(indvInstance.attributes.get(i), counts);
			}
		}
		// --------------------------------------------------------------------
		// This section will be responsible for looping through the HashMap's
		// key's (which are the classes G and B). This will be the initial 
		// start of calculating the Entropy/H values of the classes.
		// will be used as a heuristic for estimating the (relative) 
		// tree size rooted at a node given a set of examples associated 
		// with that node
		// --------------------------------------------------------------------
		// TL;DR: calculates H(Y)
		// --------------------------------------------------------------------		
		double HValue = 0;
		double total = (double) instances.size();
		Iterator iter = hashMapOfLabels.entrySet().iterator();
		while (iter.hasNext()) 
		{
			// Create a hValue entry containing the String and relevant Integer
			// value with the count of the Label.
			Map.Entry pair = (Map.Entry)iter.next();
			HValue += -((double) (int)pair.getValue()/total)
					*(Math.log((double) (int)pair.getValue()/total)/Math.log(2));
		}
		// --------------------------------------------------------------------
		// This section's loop will make one iteration through all of the 
		// remaining available attributes. If a particular attribute has 
		// already been chosen it will not reassign. Only assigns the attributes
		// which have yet to be selected. During which the conditional entropy
		// of the attribute will also be calculated and added to a list  
		// containing other conditional entropy values.
		// --------------------------------------------------------------------
		// TL;DR: calculates H(Y | X)
		// --------------------------------------------------------------------		
		List<Double> condHList = new ArrayList<Double>(); 
		for (String indvAttribute : attributes)
		{
			double condHValue = 0;
			// identify conditional entropy values (i.e. H(class | A1, A2, ...))
			int index = getAttributeIndex(indvAttribute);
			// only view/analyze attributes left in our list
			HashMap<String, List<Integer>> map = listOfAttributeHashMaps.get(index);
			iter = map.entrySet().iterator();
			// look at each attribute value
			while (iter.hasNext()) 
			{
				// ------------------------------------------------------------
				// This will create a bundled object containing the
				// value of each attribute and the total number occurrences 
				// counted, using the iterator.
				// ------------------------------------------------------------
				Map.Entry<String, List<Integer>> pair 
				= (Map.Entry<String, List<Integer>>)iter.next();
				// ------------------------------------------------------------
				// retrieves the count collection for the given attribute 
				// bundle. Index 0 of counterList represents the number of
				// times the attribute value was found. While index 1 and 2
				// represent the number of instances of class labels G and B
				// ------------------------------------------------------------
				List<Integer> counts = pair.getValue();
				for(int i = 1; i < counts.size(); i++)
				{
					// --------------------------------------------------------
					// H(Y | X) -> H(Class Label | Attribute) =
					// --------------------------------------------------------
					// (Count of Attribute 1 / total) * H(CoA1<G>/CoA1, CoA1<B>/CoA1) 
					// + (Count of Attribute 2 / total) * H(CoA2<G>/CoA2, CoA2<B>/CoA2)
					// + (Count of Attribute 3 / total) * H(CoA3<G>/CoA3, CoA3<B>/CoA3)
					// ........
					// + (Count of Attribute N/ total) * H(CoAN<G>/CoAN, CoAN<B>/CoAN)
					// --------------------------------------------------------
					if (counts.get(i) != 0)
					{
						// where the computation of each attribute occurs
						// (Count of Attribute N/ total) * H(CoAN<G>/CoAN, CoAN<B>/CoAN)
						condHValue += -((double) counts.get(i)/counts.get(0))*((double) counts.get(0)/total)
								*(Math.log((double) counts.get(i)/counts.get(0))/Math.log(2));
					}
				}
			}
			//add new conditional entropy to end of list.
			condHList.add(condHValue);
		}

		// Loop within list output largest H(Class Label) - H(Label | Attr)
		double minCondH = condHList.get(0);
		int indexOfBestAttr = 0;
		for (int i = 0; i < condHList.size(); i++)
		{
			if(minCondH > condHList.get(i))
			{
				minCondH = condHList.get(i);
				indexOfBestAttr = i;
			}
		}
		// Optimal attribute has same index in attributes as its condHValue
		// in list of conditional entropy values.
		return attributes.get(indexOfBestAttr);
	}
	
	/*
	 * Determines the "Default" value of a node used in building the Decision Tree.
	 */
		private DecTreeNodeImpl majorityVotedClass(List<Instance> instances)
		{
			int majorityLabelIndex = 0;
			int highestCount = 0;
			int tieBreakerIndex = 0;
			int currCount;
			DecTreeNodeImpl node;
			// This will go through and count the amount of each existing label
			// and update until the label with the "majority vote" or existing
			// majority is identified.
			for (String labelVar : this.labels) 
			{
				currCount = 0;
				// Loop that will count the number of instances with this label.
				for (Instance i : instances)
				{
					if (i.label.equals(labelVar))
						currCount++;
				}
				// if the label has more counts than the current maximum counted 
				// label, replace maxCount with the new value and retrieve the
				// necessary index number for the label. 
				if (currCount > highestCount)
				{
					highestCount = currCount;
					majorityLabelIndex = getLabelIndex(labelVar);
				}
				// create a new Decision Tree node that will be used as the default
				// classifier
				else if(currCount == highestCount && highestCount != 0)
				{
					tieBreakerIndex = getLabelIndex(labelVar);
					if(tieBreakerIndex < majorityLabelIndex)
					{
						majorityLabelIndex = tieBreakerIndex;
					}
				}
			}
			node = new DecTreeNodeImpl(this.labels.get(majorityLabelIndex), null, null, true);
			return node;
		}
	
	/*
	 * creates a string array that is used to compare against in order to 
	 * determine accuracy metrics.
	 */
	private String[] pipeResults(DataSet test) 
	{
		String[] results = new String[test.instances.size()];

		for (int i = 0; i < test.instances.size(); i++) 
		{
			results[i] = classify(test.instances.get(i));
		}

		return results;
	}
	
	/*
	 * checks against the array created in the pipeResults method
	 * and uses that string array vs. the actual label. If correct,
	 * adds to the correctness evaluation.
	 */
	private double determineAccuracy(DataSet test) 
	{
		String results[] = pipeResults(test);
		double correct = 0.0;
		double total = test.instances.size();
		for (int i = 0; i < test.instances.size(); i ++)
			if (test.instances.get(i).label.equals(results[i]))
				correct ++;
		return correct / total;
	}
	
	/*
	 * travels throughout the tree, starting from the root node.
	 * used to sift through the tree and actively prune/build the decision tree
	 * using values calculated prior in the assignment. 
	 */
	private void traverseTree(DecTreeNodeImpl node, ArrayList<DecTreeNodeImpl> nodeListingFromRoot) 
	{
		for (DecTreeNode child: node.children) 
		{
			if (!child.terminal) 
			{
				DecTreeNodeImpl tempChild = (DecTreeNodeImpl) child;
				traverseTree(tempChild, nodeListingFromRoot);
			}
		}
		nodeListingFromRoot.add(node);
	}

}

