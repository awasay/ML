from sklearn import linear_model
from sklearn.svm import SVC
from sknn.mlp import Classifier, Layer
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
import csv

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


# t is the test_id if predict is true or t_valid if predict is false

## Logistic regression

def LogReg(X_train,t_train,x,t,predict):

	logreg = linear_model.LogisticRegression()

	logreg.fit(X_train, t_train)
	prediction = logreg.predict(x)

	if predict:
		write_predictions(t,prediction)
	else:
		get_accuracy(prediction,t)

## Linear SVM

def LSVM(X_train,t_train,x,t,predict):

	svc = SVC(kernel="linear")


	svc.fit(X_train, t_train)
	prediction = svc.predict(x)

	if predict:
		write_predictions(t,prediction)
	else:
		get_accuracy(prediction,t)

## Neural Network

def NN(X_train,t_train,x,t,predict):

	clf = ExtraTreesClassifier(n_estimators=500, max_depth=None)

	clf.fit(X_train, t_train)
	prediction = clf.predict(x)

	if predict:
		write_predictions(t,prediction)
	else:
		get_accuracy(prediction,t)
