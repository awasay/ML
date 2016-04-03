from sklearn import linear_model
from sklearn.svm import SVC
from sknn.mlp import Classifier, Layer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import ensemble

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

    print prediction
    print test_ids
    
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

	svc = SVC(gamma=3,C=1)


	svc.fit(X_train, t_train)
	prediction = svc.predict(x)

	if predict:
		write_predictions(t,prediction)
	else:
		get_accuracy(prediction,t)

## Neural Network

def EXRT(X_train,t_train,x,t,predict):
	for i in [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]:	
		clf = ExtraTreesClassifier(n_estimators=500, max_depth=None, max_features=i)

		clf.fit(X_train, t_train)
		prediction = clf.predict(x)
		if predict:
			write_predictions(t,prediction)
		else:
			get_accuracy(prediction,t)

def gb(X_train,t_train,x,t,predict):

	for i in [10]:
		params = {'n_estimators': 500, 'learning_rate': 0.01, "max_features":i}
		clf = ensemble.GradientBoostingClassifier(**params)
		
		clf.fit(X_train, t_train)
		prediction = clf.predict(x)
		print i
		if predict:
			write_predictions(t,prediction)
		else:
			get_accuracy(prediction,t)


def adaboost(X_train,t_train,x,t,predict):
	clf = AdaBoostClassifier()
	clf.fit(X_train, t_train)
	prediction = clf.predict(x)

	if predict:
		write_predictions(t,prediction)
	else:
		get_accuracy(prediction,t)
