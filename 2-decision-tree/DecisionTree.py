import numpy as np
from collections import Counter
class Node():
    """
    A class representing a node in a decision tree.
    """

    def __init__(self, feature=None, threshold=None, left=None, right=None, gain=None, value=None):
        """
        Initializes a new instance of the Node class.

        Args:
            feature: The feature used for splitting at this node. Defaults to None.
            threshold: The threshold used for splitting at this node. Defaults to None.
            left: The left child node. Defaults to None.
            right: The right child node. Defaults to None.
            gain: The gain of the split. Defaults to None.
            value: If this node is a leaf node, this attribute represents the predicted value
                for the target variable. Defaults to None.
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.gain = gain
        self.value = value
class DecisionTree:
    """
    A decision tree classifier for binary classification problems.
    """
    def __init__(self, min_samples = 2 ,max_depth = 2):
        """
        Constructor for DecisioTree class.
        
        Parameters:
            min_samples (int): Minimum number of samples required to split an internal node.
            max_depth (int): Maximum depth of the decisioin tree.
        """

        self.min_samples = min_samples
        self.max_depth = max_depth
    
    def split_data(self, dataset, feature, threshold):
        """
        Splits the given dataset into two datasets based on the given freature and threshold.
        
        Parameters:
            dataset (ndarray) : Input dataset.
            feature (int): Index of the feature to be split on.
            threshold (float): Threshold value to split it the feature on.

            Return:
                left_dataset (nparray): Subset of the dataset with values less than or equal to the threshold.
                right_dataset (ndarray): Subset of the dataset with values greater than the threshold.
        """

        # Create dmpty arrays to store te left and right datasets
        left_dataset = []
        right_dataset = []

        #Loop over each row in the dataset and split based on the given feature and threshold
        for row in dataset:
            if row[feature] <= threshold:
                left_dataset.append(row)
            else:
                right_dataset.append(row)
        
        # Convert the left and right dataset to numpy arrays and return
        left_dataset = np.array(left_dataset)
        right_dataset = np.array(right_dataset)
        return left_dataset, right_dataset
    
    def entropy(self, y):
        """
        Computes the entropy of the given label values.

        Paremetes:
            y (ndarray): Input label values.

        Returns:
            entropy (float): Entropy of the given lable values.
        """

        # fomula entropy = - sum( log(p) * p )  with p is ratio of feature in the dataset
        # function bit count caculate the numbers of features in the data set y
        hist = np.bincount(y) 
        ps = hist/len(y)
        return - np.sum([p * np.log(p) for p in ps if p > 0])
    
    def information_gain(self, parent, left, right):
        """
        Computers the information gain from splitting the parent dataset.

        Parameters:
            parrent (ndarray): Input parent dataset.
            left (ndarray): Subset of the parent data set after split on a feature.
            right  (ndarray): Subset of the parent data set after split on a feature.

            Returns:
                information_gain (float): Informatioin gain of the split
        """
        #set initial information gain to 0
        information_gain = 0
        #compute entropy for parent
        parent_entropy = self.entropy(parent)

        # calculate weight for left and right nodes
        weight_left = len(left) / len(parent)
        weight_right = len(right)/ len(parent)
        # copute entropy for left and right nodes
        entropy_left , entropy_right = self.entropy(left),self.entropy(right)
        # calculate weighted entropy
        weighted_entropy = weight_left*entropy_left + weight_right * entropy_right
        #calculate information gain
        information_gain = parent_entropy - weighted_entropy 
        return information_gain