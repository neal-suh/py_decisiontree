# Import pandas, numpy, matplotlib, and sklearn
import pandas as pd # Used to create decision tree from scratch
import numpy as np # Used to create decision tree from scratch
from matplotlib import pyplot as plt # Only used to verify results
from sklearn import tree # Only used to verify results
from sklearn.tree import DecisionTreeClassifier as sklearnDTC # Only used to verify results
from sklearn.tree import DecisionTreeRegressor as SklearnDTR # Only used to verify results

# Define minimizing functions: entropy, gini, variance
def entropy(y):
    if y.size == 0: return 0
    p = np.unique(y, return_counts = True)[1].astype(float) / len(y)
    return -1 * np.sum(p * np.log2(p + 1e-9))

def gini(y):
    if y.size == 0: return 0
    p = np.unique(y, return_counts = True)[1].astype(float) / len(y)
    return 1 - np.sum(p**2)

def variance(y):
    if y.size == 0: return 0
    return np.var(y)



# Decision Tree Node
class DTNode(object):
    
    # Initialize all default parameters
    def __init__(self, X, y, minimize_func, min_info_gain = 0.001, max_depth = 3, min_leaf_size = 20, depth = 0):
        self.X = X
        self.y = y
        self.minimize_func = minimize_func
        self.min_info_gain = min_info_gain
        self.min_leaf_size = min_leaf_size
        self.max_depth = max_depth
        self.depth = depth
        self.best_split = None
        self.left = None
        self.right = None
        self.is_leaf = True
        self.split_description = "root"
        
    # Call information gain function if entropy is the metric
    def info_gain(self, mask):
        s1 = np.sum(mask)
        s2 = mask.size - s1
        
        if (s1 == 0 | s2 == 0):
            return 0
        
        return self.minimize_func(self.y) - s1 / float(s1 + s2) * self.minimize_func(self.y[mask]) - s2 / float(s1 + s2) * self.minimize_func(self.y[np.logical_not(mask)])
    
    # Search for the best split among all the values in a given feature
    def max_info_gain_split(self, x):
        best_change = 0
        split_value = 0
        previous_val = 0
        is_numeric = x.dtype.kind not in ['S','b']

        for val in np.unique(np.sort(x)):
            mask = x == val
            
            if (is_numeric):
                mask = x < val
                
            change = self.info_gain(mask)
            s1 = np.sum(mask)
            s2 = mask.size - s1
            
            if best_change == 0 and s1 >= self.min_leaf_size and s2 >= self.min_leaf_size:
                best_change = change
                split_value = val
            elif change > best_change and s1 >= self.min_leaf_size and s2 >= self.min_leaf_size:
                best_change = change
                split_value = np.mean([val,previous_val])
            
            previous_val = val

        return {"best_change":best_change, "split_value":split_value, "is_numeric":is_numeric}
    
    # Return the information for the best split
    def best_feat_split(self):
        best_result = None
        best_feat = None
        
        for feat in range(self.X.shape[1]):
            result = self.max_info_gain_split(self.X[:,feat])
            
            if result['best_change'] is not None:
                if best_result is None:
                    best_result = result
                    best_feat = feat
                elif best_result['best_change'] < result['best_change']:
                    best_result = result
                    best_feat = feat
        
        if best_result is not None:
            best_result['feat'] = best_feat
            self.best_split = best_result
    
    # Split into left and right branches
    def split_node(self):
        if self.depth < self.max_depth:
            self.best_feat_split() 
            
            if self.best_split is not None and self.best_split['best_change'] >= self.min_info_gain:
                mask = None
                
                if self.best_split['is_numeric']:
                    mask = self.X[:,self.best_split['feat']] < self.best_split['split_value']
                else:
                    mask = self.X[:,self.best_split['feat']] == self.best_split['split_value']
                
                if(np.sum(mask) >= self.min_leaf_size and (mask.size - np.sum(mask)) >= self.min_leaf_size):
                    self.is_leaf = False
                    self.left = DTNode(self.X[mask,:], self.y[mask], self.minimize_func, self.min_info_gain, self.max_depth, self.min_leaf_size, self.depth+1)

                    if self.best_split['is_numeric']:
                        split_description = ' [feature ' + str(self.best_split['feat']) + "] < " + str(self.best_split['split_value']) + " (" + str(self.X[mask,:].shape[0]) + ")"
                        self.left.split_description = str(split_description)
                    else:
                        split_description = ' [feature ' + str(self.best_split['feat']) + "] == " + str(self.best_split['split_value']) + " (" + str(self.X[mask,:].shape[0]) + ")"
                        self.left.split_description = str(split_description)

                    self.left.split_node()
                    self.right = DTNode(self.X[np.logical_not(mask),:], self.y[np.logical_not(mask)], self.minimize_func, self.min_info_gain, self.max_depth, self.min_leaf_size, self.depth + 1)
                    
                    if self.best_split['is_numeric']:
                        split_description = ' [feature ' + str(self.best_split['feat']) + "] >= " + str(self.best_split['split_value']) + " (" + str(self.X[np.logical_not(mask),:].shape[0]) + ")"
                        self.right.split_description = str(split_description)
                    else:
                        split_description = ' [feature ' + str(self.best_split['feat']) + "] != " + str(self.best_split['split_value']) + " (" + str(self.X[np.logical_not(mask),:].shape[0]) + ")"
                        self.right.split_description = str(split_description)

                    self.right.split_node()
                    
        if self.is_leaf:
            if self.minimize_func == variance:
                self.split_description = self.split_description + " : predict - " + str(np.mean(self.y))
            else:
                values, counts = np.unique(self.y,return_counts = True)
                predict = values[np.argmax(counts)]
                self.split_description = self.split_description + " : predict - " + str(predict)
                                          
    # Make a prediction for every row
    def predict_row(self, row):
        predict_value = None
        
        if self.is_leaf:
            if self.minimize_func == variance:
                predict_value = np.mean(self.y)
            else:
                values, counts = np.unique(self.y,return_counts = True)
                predict_value = values[np.argmax(counts)]
        else:
            left = None
            if self.best_split['is_numeric']:
                left = row[self.best_split['feat']] < self.best_split['split_value']
            else:
                left = row[self.best_split['feat']] == self.best_split['split_value']
                
            if left:
                predict_value = self.left.predict_row(row)
            else:
                predict_value = self.right.predict_row(row)
 
        return predict_value
    
    # Predict the values from the fitted tree
    def predict(self, X):
        return np.apply_along_axis(lambda x: self.predict_row(x), 1, X)
    
    
    def rep(self, level):
        response = "o--> " + self.split_description
        
        if self.left is not None:
            response += "\n" + (2 * level + 1) * " " + self.left.rep(level + 1)
        if self.right is not None:
            response += "\n" + (2 * level + 1) * " " + self.right.rep(level + 1)
        
        return response
    
    
    def __repr__(self):
        return self.rep(0)
    


# Decision Tree
class DecisionTree(object):
    
    # Initialize all default parameters
    def __init__(self, minimize_func, min_info_gain = 0.001, max_depth = 3, min_leaf_size = 20):
        self.root = None
        self.minimize_func = minimize_func
        self.min_info_gain = min_info_gain
        self.max_depth = max_depth
        self.min_leaf_size = min_leaf_size
        
    # Take features and outputs
    def fit(self, X, y):
        self.root =  DTNode(X, y, self.minimize_func, self.min_info_gain, self.max_depth, self.min_leaf_size, 0)
        self.root.split_node()
    
    # Wrapper to call predict function in DTNode
    def predict(self,X):
        return self.root.predict(X)
    
    
    def __repr__(self):
        return self.root.rep(0)