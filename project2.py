from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve, auc
from operator import itemgetter
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np
import itertools
import sys
import math
import re

if len(sys.argv) < 3:
	print "Enter: <filename> <metricType>"
	sys.exit()

filename = sys.argv[1]
metricType = sys.argv[2]

def isNominal(val):
	return isinstance(val, basestring)

def gowerSimilarity(rec1, rec2):
	# nominal: equal=1, else 0
	# manhattan distance based similarity for ordinal/ratio
	sim = 0
	for val1,val2 in zip(rec1,rec2):
		if isNominal(val1):
			sim = sim + (val1==val2)
		else:
			d = math.fabs(val1-val2)
			sim = sim + (1/(1+d))
	return sim / len(rec1)

def manhattanDistance(rec1,rec2):
	try:
		dist = sum(math.fabs(val1-val2) for (val1,val2) in zip(rec1,rec2))
	except:
		dist  = 0
		for (val1,val2) in zip(rec1,rec2):
			try:
				dist += math.fabs(val1-val2)
			except:
				print val1, val2
				sys.exit()

	return sum(math.fabs(val1-val2) for (val1,val2) in zip(rec1,rec2))

def euclideanDistance(rec1, rec2):
	return math.sqrt(sum(math.pow(val1 - val2,2) for (val1,val2) in zip(rec1,rec2)))

def getNeighboursIris(trainingData, testRecord, metricType, k=5):
	# Returns (prox measure,indices) of k nearest neighbours in training set 
	values = []
	for (index,row) in trainingData.iterrows():
		# Choose comparison metric (distance/similarity)
		# modify desireHighValue False - distance, True - similarity
		if metricType == "euclidean":
			metric = euclideanDistance(testRecord, row)
			desireHighValue = False
		elif metricType == "manhattan":
			metric = manhattanDistance(testRecord, row)
			desireHighValue = False

		values.append((metric, index))
	values.sort(key = itemgetter(0), reverse = desireHighValue)

	# values :: (distance,ind)
	return values[0:k]


def getNeighboursIncome(trainingData, testRecord, metricType, k = 5):
    values = []
    for (index,row) in trainingData.iterrows():
		# Choose comparison metric (distance/similarity)
		# modify desireHighValue False - distance, True - similarity
        if metricType == "gower":
        	metric = gowerSimilarity(testRecord, row)
        	desireHighValue = True
        	metric = 1/metric
        	desireHighValue = False
        elif metricType == "binarizedManhattan":
            metric = manhattanDistance(testRecord.values, row.values)
            desireHighValue = False

        values.append((metric, index))
    values.sort(key = itemgetter(0), reverse = desireHighValue)

    return values[0:k]

def preprocessIncomeData(data, metricType, countries=None, isAgeBin='N',isCountryBinary='N'):

	data['capital_gain'] = (data['capital_gain'] > 0).astype(int)
	data['capital_loss'] = (data['capital_loss'] > 0).astype(int)

	# Obs. for large number of records native_country = United-States
	data.loc[data["native_country"].str.contains('^\s*\?$'),"native_country"] = "United-States"
	# Obs. for large number of records wokrclass = Private
	data.loc[data["workclass"].str.contains('^\s*\?$'),"workclass"] = "Private"

	# Education = education_cat (redundancy)
	# ID = irrelevant
	# occupation = several missing values, no majority value thus, no appropriate way of filling in
	data = data.drop(['occupation','education','ID'],1)

	numCols = []
	stringCols = []

	for col in data.columns:
	    if(data[col].dtype == np.float64 or data[col].dtype == np.int64):
	    	numCols.append(col)
	    else:
	    	data[col] = data[col].str.strip()
	    	stringCols.append(col)


	# Group workclass into private, gov, self-emp
	selfEmp = {'Self-emp-not-inc','Self-emp-inc'}
	gov = {'Local-gov', 'State-gov', 'Federal-gov'}
	for (i,c) in data['workclass'].iteritems():
		if c in selfEmp:
			data.loc[i,'workclass'] = 'Self-emp'
		elif c in gov:
			data.loc[i,'workclass'] = 'Gov'

	if isAgeBin=='Y':
		bins = [0, 25, 35, 45, 55, 65, 75, 85, 120]
		group_names = [2,3,4,5,6,7,8,9]
		categories = pd.cut(data['age'], bins, labels=group_names)
		data['age'] = categories

	# Normalize numeric ratio data and ordinal
	cols_to_norm = numCols
	data[cols_to_norm] = data[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

	if isCountryBinary=='Y':
		for (i,r) in data.iterrows():
			if r['native_country'] == 'United-States':
				data.loc[i,'native_country'] = 1
			else:
				data.loc[i,'native_country'] = 0

	if metricType=='binarizedManhattan':
		# Convert nominal attributes to dummy binary variables
		data_new = data[numCols]
		for col in stringCols:
			if isCountryBinary=='Y' and col == 'native_country':
				data_new['native_country'] = data['native_country']
				continue
			if col == 'class':
				continue
			data_new = pd.concat([data_new,pd.get_dummies(data[col], prefix=col)], axis = 1)
		
		data_new['class'] = data['class']
		data = data_new

		# Combine Other Countries not present in Training Data
		if isCountryBinary=='N' and countries is not None:
			countryCols = data.columns[data.columns.str.contains('^native_country_')]
			
			otherCountries = set(countryCols) - set(countries)
			data_new['native_country_ Other'] = 7
			
			for (i,r) in data.iterrows():
				if len(otherCountries)!=0:
					data.loc[i,'native_country_ Other'] = int(max(data.loc[i,list(otherCountries)].values))
				else:
					data.loc[i,'native_country_ Other'] = 0
			data = data.drop(otherCountries,1)
		elif isCountryBinary=='N':
			data['native_country_ Other'] = 0

	return data

if filename=="Iris_Test.csv" or filename=='Iris.csv':
	if (metricType!="euclidean" and (metricType!="manhattan")):
		print "Choose metric type from [euclidean/manhattan]"
		sys.exit()

	trainFileName = 'Iris.csv'
	testFileName = filename

	trainData = pd.read_csv(trainFileName)
	testData = pd.read_csv(testFileName)

	trainData.columns = trainData.columns.str.strip()
	testData.columns = testData.columns.str.strip()

	X_train = trainData.as_matrix(columns=trainData.columns[:-1])
	Y_train = trainData.as_matrix(columns=[trainData.columns[-1]])
	X_test = testData.as_matrix(columns=testData.columns[:-1])
	Y_test = testData.as_matrix(columns=[testData.columns[-1]])

	k = int(raw_input("Enter value of k: ") or "5") 
	print(k)

	weights = 'distance' if (raw_input("Weighted Classifier?(Y/N)") or 'N').upper()=='Y' else 'uniform'

	# Test on own classifier
	rowList = []
	for (ind, record) in testData.iterrows():
		newRec = {}
		actualClass = record[-1]
		neighbors = getNeighboursIris(trainData.drop('class',1), record, metricType, k)

		if weights == 'uniform':
			votes = []
			for i in range(k):
				votes.append(trainData.iloc[neighbors[i][1]]['class'])
			count = Counter(votes)
			predClass = count.most_common(1)[0][0]

			prob = float(count.most_common(1)[0][1]) / k

		else:
			votes = {}
			for (dist, ind) in neighbors:
				cls = trainData.iloc[ind]['class']
				if cls in votes:
					votes[cls] += 1/(dist+1)
				else:
					votes[cls] = 1/(dist+1)
			predClass = max(votes.iteritems(), key=itemgetter(1))[0]
			prob = max(votes.iteritems(), key=itemgetter(1))[1] / sum(votes.values())

		newRec['Transaction'] = ind
		newRec['Actual Class'] = actualClass
		newRec['Predicted Class'] = predClass
		newRec['Posterior Probability'] = prob
		rowList.append(newRec)

	df_ = pd.DataFrame(rowList)
	df_.to_csv("iris_knn_" + metricType + "_out_" + weights + ".csv")

elif filename=='income_te.csv' or filename=='income_tr.csv':

	if (metricType!="gower") and (metricType!="binarizedManhattan"):
		print "Choose metric type from [gower/binarizedManhattan]"
		sys.exit()

	trainFileName = 'income_tr.csv'
	testFileName = filename

	toSkip = (raw_input("Skip rows in training data?(Y/N)") or 'N').upper()
	if toSkip=='Y':
		n = 520
		s = n/2
		skip = sorted(random.sample(xrange(1,n),n-s))
	else:
		skip = 0
	trainData = pd.read_csv(trainFileName, skiprows=skip)
	testData = pd.read_csv(testFileName)

	trainData.columns = trainData.columns.str.strip()
	testData.columns = testData.columns.str.strip()

	k = int(raw_input("Enter value of k: ") or "5") 
	print(k)

	weights = 'distance' if (raw_input("Weighted Classifier?(Y/N)") or 'N').upper()=='Y' else 'uniform'

	# Preprocessing
	isAgeBin = (raw_input("Divide Age into bins?(Y/N)") or 'N').upper()
	isCountryBinary = (raw_input("Binarize country as US/Non-US?(Y/N)") or 'N').upper()

	trainData = preprocessIncomeData(trainData, metricType, isAgeBin=isAgeBin, isCountryBinary=isCountryBinary)
	listCountries = trainData.columns[trainData.columns.str.contains('^native_country_')]
	testData = preprocessIncomeData(testData, metricType, countries=listCountries, isAgeBin=isAgeBin, isCountryBinary=isCountryBinary)

	extraColsInTraining = set(trainData.columns) - set(testData.columns)
	for c in extraColsInTraining:
		testData[c] = 0

	# Make sure the columns are in same order in train and test dfs
	colsTrain = sorted(trainData.columns.tolist())
	colsTest = sorted(testData.columns.tolist())

	trainData = trainData[colsTrain]
	testData = testData[colsTest]

	trainAttributes = trainData.columns[trainData.columns!='class']
	X_train = trainData[trainAttributes].as_matrix()
	Y_train = trainData['class'].values
	X_test = testData[trainAttributes].as_matrix()
	Y_test = testData['class'].values

	# Test on own classifier
	rowList = []
	for (ind, record) in testData.iterrows():
		newRec = {}
		actualClass = record['class']
		#####
		neighbors = getNeighboursIncome(trainData.drop('class',1), record[trainAttributes], metricType, k)
		#####
		if weights == 'uniform':
			votes = []
			for i in range(k):
				votes.append(trainData.iloc[neighbors[i][1]]['class'])
			count = Counter(votes)
			predClass = count.most_common(1)[0][0]

			prob = float(count.most_common(1)[0][1]) / k

		else:
			votes = {}
			for (dist, ind) in neighbors:
				cls = trainData.iloc[ind]['class']
				if cls in votes:
					votes[cls] += 1/(dist+1)
				else:
					votes[cls] = 1/(dist+1)
			predClass = max(votes.iteritems(), key=itemgetter(1))[0]
			prob = max(votes.iteritems(), key=itemgetter(1))[1] / sum(votes.values())

		newRec['Transaction'] = ind
		newRec['Actual Class'] = actualClass
		newRec['Predicted Class'] = predClass
		newRec['Posterior Probability'] = prob
		rowList.append(newRec)

	df_ = pd.DataFrame(rowList)
	df_.to_csv("income_knn_" + metricType + "_out_" + weights + ".csv")
	

else:
	print 'Choose filename from [income_te.csv|Iris_Test.csv|income_te.csv|Iris.csv]'
	sys.exit()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Confusion matrix")
    else:
        print('Confusion matrix')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


class_names = trainData['class'].unique()
print
print "== Test on own " + metricType + " based KNN Classifier =="
print classification_report(df_['Actual Class'], df_['Predicted Class'])
print 'Accuracy', accuracy_score(df_['Actual Class'], df_['Predicted Class'])
cnf_matrix = confusion_matrix(df_['Actual Class'],df_['Predicted Class'])
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, 
                      title='Confusion matrix')
plt.show()

dummyClass = pd.get_dummies(df_['Actual Class'], prefix='class')


if filename in {'income_te.csv','income_tr.csv'}:
	i=0
	fpr, tpr, _ = roc_curve(dummyClass[dummyClass.columns[i]], df_['Posterior Probability'])
	roc_auc = auc(fpr, tpr)
	plt.figure()
	lw = 2
	plt.plot(fpr, tpr, color='darkorange',
	         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic')
	plt.legend(loc="lower right")
	plt.show()

if metricType!='gower':
	print
	print "== Test on sklearn KNeighborsClassifier =="
	# Test on sklearn knn classifier
	for k in range(1,30):
		knn = KNeighborsClassifier(n_neighbors=k, weights = weights)
		knn.fit(X_train, Y_train.ravel()) 
		predY = knn.predict(X_test)
		#print classification_report(Y_test, predY)
		print k,'Accuracy', accuracy_score(Y_test, predY)
	cnf_matrix = confusion_matrix(Y_test,predY)
	plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=class_names, 
	                      title='Confusion matrix, sklearn KNN')

	plt.show()



