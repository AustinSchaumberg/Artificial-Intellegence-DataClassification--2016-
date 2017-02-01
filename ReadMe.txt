Austin Schaumberg HW3 ReadMe.txt File
*** PRUNING ALGORITHM WAS COMPLETED, PLEASE TEST OR GIVE OVERVIEW***

About my constructor DecisionTreeImpl(DataSet train, DataSet tune) and its results:
Methods used to accomplish pruning:
traverseTree, determineAccuracy, pipeResults

// Args used for outputs: 4 prune_train.txt prune_test.txt prune_tune.txt//
////////////// First 30 lines of Pruned DataSet///////////
ROOT {A1?}
    x {A3?}
        a {A5?}
            h (B)
            s (G)
            n (G)
            u (G)
        c {A2?}
            n {A5?}
                h {A4?}
                    r (G)
                    o (G)
                    f (B)
                s (G)
                n (G)
                u {A4?}
                    r (G)
                    o (B)
                    f (G)
            b {A6?}
                c {A5?}
                    h {A8?}
                        1 (G)
                        2 {A9?}
                            y {A4?}
                                r (G)
                                o {A7?}
                                    1 {A10?}
                                        y (G)
                                        n (G)
                                    2 (G)
////////////// First 30 lines of Pruned DataSet ///////////

// Args used for outputs: 5 prune_train.txt prune_test.txt prune_tune.txt//
////////////// First 30 lines of Pruned DataSet Class labels///////////
B
B
G
G
G
G
G
G
B
G
G
B
B
B
G
G
B
G
G
G
G
G
G
G
G
G
G
B
G
B
G
G
G

// Args used for outputs: 6 prune_train.txt prune_test.txt prune_tune.txt//
///////////////////////////////Output////////////////////////////////////
 0.68098
 // Comments on accuracy. 
 These make sense, as my K-3 folds accuracy of the dataset also lead me to almost this exact same value.


 ////////////////////// About The Pruning Decision Tree Method://///////////////////

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
///////////////////////// The Private Methods: ///////////////////////// 
 	/*
	 * creates a string array that is used to compare against in order to 
	 * determine accuracy metrics.
	 */
	public String[] pipeResults(DataSet test) 
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
	 * used to sift through the tree and actively prune the decision tree. 
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

