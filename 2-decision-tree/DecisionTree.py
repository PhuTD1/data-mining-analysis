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
    def __init__(self, min_samples = 2, max_depth = 2):
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

        # Loop over each row in the dataset and split based on the given feature and threshold
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
        # set initial information gain to 0
        information_gain = 0
        # compute entropy for parent
        parent_entropy = self.entropy(parent)

        # calculate weight for left and right nodes
        weight_left = len(left) / len(parent)
        weight_right = len(right) / len(parent)
        # copute entropy for left and right nodes
        entropy_left, entropy_right = self.entropy(left), self.entropy(right)
        # calculate weighted entropy
        weighted_entropy = weight_left*entropy_left + weight_right * entropy_right
        # calculate information gain
        information_gain = parent_entropy - weighted_entropy 
        return information_gain
    def best_split(self, dataset, num_samples, num_features):
        """
        Finds the best split for the given dataset.
        Args:
        dataset (ndarray): The dataset to split.
        num_samples (int): The number of samples in the dataset.
        num_features (int): The number of features in the dataset.

        Returns:
        dict: A dictionary with the best split feature index, threshold, gain,
            lef and right dataset
        """
        # dictionary to stores the best split values
        best_split = {'gain': -1, 'feature': None, 'threshold': None}
        # loop over all the features
        for feature_idx in range(num_features):
            # get the feature at the current feature_index
            feature_values = dataset[:, feature_idx]
            # get unique values of that feature
            thresholds = np.unique(feature_values)
            # loop over all values fo the feature
            for threshold in thresholds:
                # get left and right dataset
                left_dataset, right_dataset = self.split_data(dataset, feature_idx, threshold)
                # check if either datset is empty
                if len(left_dataset) and len(right_dataset):
                    # get y values of the parent and left, right nodes
                    y, left_y, right_y = dataset[:, -1], left_dataset[:, -1], right_dataset[:, -1]
                    # compute informationi gain based on the y values
                    information_gain = self.information_gain(y, left_y, right_y)
                    # update the best split if conditions are met
                    if information_gain > best_split["gain"]:
                        best_split["feature"] = feature_idx
                        best_split["threshold"] = threshold
                        best_split["left_dataset"] = left_dataset
                        best_split["right_dataset"] = right_dataset
                        best_split["gain"] = information_gain
        return best_split
