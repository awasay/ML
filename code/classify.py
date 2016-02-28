# Example Feature Extraction from XML Files
# We count the number of specific system calls made by the programs, and use
# these as our features.

# This code requires that the unzipped training set is in a folder called "train". 

# Wasay: I have extended the sample code. Comments within the file.

import os
from collections import Counter
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import numpy as np
from scipy import sparse
import csv

import util

TRAIN_DIR = "../train"
TEST_DIR = "../test"
CALLS = {}

call_set = set([])

def add_to_set(tree):
    for el in tree.iter():
        call = el.tag
        call_set.add(call)

# Wasay: Extract a set of unique sys calls in the entire training data set:

def get_unique_calls(start_index, end_index, direc="train"):
    X = None
    classes = []
    ids = [] 
    i = -1
    for datafile in os.listdir(direc):
        if datafile == '.DS_Store':
            continue

        i += 1
        if i < start_index:
            continue 
        if i >= end_index:
            break

        # extract id and true class (if available) from filename
        id_str, clazz = datafile.split('.')[:2]
        ids.append(id_str)
        # add target class if this is training data
        try:
            classes.append(util.malware_classes.index(clazz))

        except ValueError:
            # we should only fail to find the label in our list of malware classes
            # if this is test data, which always has an "X" label
            assert clazz == "X"
            classes.append(-1)

        # parse file as an xml document
        tree = ET.parse(os.path.join(direc,datafile))
        add_to_set(tree)
        this_row = call_feats(tree)

        # Accumulate all the unique calls in CALLS
        for el in tree.iter():
            call = el.tag
            if call not in CALLS:
                CALLS[call]=1
            else:
                CALLS[call]+=1
        

def create_data_matrix(start_index, end_index, direc="train"):
    X = None
    classes = []
    ids = [] 
    i = -1
    for datafile in os.listdir(direc):
        if datafile == '.DS_Store':
            continue

        i += 1
        if i < start_index:
            continue 
        if i >= end_index:
            break

        # extract id and true class (if available) from filename
        id_str, clazz = datafile.split('.')[:2]
        ids.append(id_str)
        # add target class if this is training data
        try:
            classes.append(util.malware_classes.index(clazz))

        except ValueError:
            # we should only fail to find the label in our list of malware classes
            # if this is test data, which always has an "X" label
            assert clazz == "X"
            classes.append(-1)

        # parse file as an xml document
        tree = ET.parse(os.path.join(direc,datafile))
        add_to_set(tree)
        this_row = call_feats(tree)
        
        if X is None:
            X = this_row 
        else:
            X = np.vstack((X, this_row))

    return X, np.array(classes), ids

# Wasay: This is the function that you can augment to extract other features:

def call_feats(tree):
    # Wasay: I am using all unique calls instead of just a subset of them.
    good_calls = CALLS.keys()

    call_counter = {}
    for el in tree.iter():
        call = el.tag
        if call not in call_counter:
            call_counter[call] = 0
        else:
            call_counter[call] += 1

    call_feat_array = np.zeros(len(good_calls))
    for i in range(len(good_calls)):
        call = good_calls[i]
        call_feat_array[i] = 0
        if call in call_counter:
            call_feat_array[i] = call_counter[call]
    return call_feat_array

# Wasay: Tests the accuracy of the prediction on validation data set. 

def get_accuracy(prediction,actual):
    assert(len(prediction)==len(actual))
    
    correct = 1.0*np.count_nonzero(prediction==actual)
    total = 1.0*(len(prediction))
    
    print "The accuracy is: "+str(correct/total)
    return correct/total

# Wasay: When you have test_ids and predictions, this function writes them to 
## result.csv in the required format.

def write_predictions(test_ids,prediction):
    assert(len(prediction)==len(test_ids))
    
    writer = csv.writer(open("../result.csv",'wb'))
    
    writer.writerow(["Id","Prediction"])
    for i in range(0,len(test_ids)):
        writer.writerow([test_ids[i],prediction[i]])

## Feature extraction
def main():

    ##
    predict = True
    # Wasay: When predict is true, we use the test data set and make actual 
    ## predictions and write them down to result.csv. When predict is false, 
    ### we divide the train data set into two halves and train on one half 
    #### and cross validate on the other. We print the accuracy.

    get_unique_calls(0, 5000, TRAIN_DIR)

    if not predict:
        X_train, t_train, train_ids = create_data_matrix(0, 1500, TRAIN_DIR)
        X_valid, t_valid, valid_ids = create_data_matrix(1500, 5000, TRAIN_DIR)

        print 'Data matrix (training set):'
        print X_train.shape
        print 'Classes (training set):'
        print t_train.shape

        from sklearn import linear_model
        # From here, you can train models (eg by importing sklearn and inputting X_train, t_train).
        
        # Logistic Regression

        logreg = linear_model.LogisticRegression()
        logreg.fit(X_train, t_train)
        prediction = logreg.predict(X_valid)

        get_accuracy(prediction,t_valid)
    else:
        X_train, t_train, train_ids = create_data_matrix(0, 5000, TRAIN_DIR)
        X_test, t_test, test_ids = create_data_matrix(0, 5000, TEST_DIR)

        print 'Data matrix (training set):'
        print X_train.shape
        print 'Classes (training set):'
        print t_train.shape

        from sklearn import linear_model
        # From here, you can train models (eg by importing sklearn and inputting X_train, t_train).
        
        # Logistic Regression

        logreg = linear_model.LogisticRegression()
        logreg.fit(X_train, t_train)
        prediction = logreg.predict(X_test)

        write_predictions(test_ids,prediction)

if __name__ == "__main__":
    main()
    