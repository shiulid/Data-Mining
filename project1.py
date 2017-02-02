import csv
import sys
import pandas as pd
import numpy as np
import math
from operator import itemgetter
from scipy import spatial
import re
import matplotlib.pyplot as plt

if len(sys.argv) < 3:
	print "Enter: <datasetfilename> <metricType>"
	sys.exit()

filename = sys.argv[1]
metricType = sys.argv[2]

try:
	data = pd.read_csv(filename)
except IOError:
	print("Couldn't open file " + filename + "!")
	sys.exit()

dataSize = len(data.index)

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
	return sum(math.fabs(val1-val2) for (val1,val2) in zip(rec1,rec2))

def euclideanDistance(rec1, rec2):
	return math.sqrt(sum(math.pow(val1 - val2,2) for (val1,val2) in zip(rec1,rec2)))

if filename=="Iris.csv":
	if (metricType!="euclidean" and (metricType!="manhattan")):
		print "Choose metric type from [euclidean/manhattan]"
		sys.exit()

	k = int(raw_input("Enter value of k: ") or "5") 
	print(k)

	data.columns = data.columns.str.strip()

	data = data.drop('class',1)

	columns = ['Transaction']
	for i in range(k):
		columns.append(str(i+1))
		columns.append(str(i+1) + "-prox")

	# Normalize data
	data.columns = data.columns.str.strip()
	cols_to_norm = ['sepal_length','sepal_width','petal_length','petal_width']
	data[cols_to_norm] = data[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
	
	# Creating Dataframe with Proximity measures
	rowList = []
	for (index1,row1) in data.iterrows():
		values = []
		selectedRow = row1
		for (index2,row2) in data.iterrows():
			if index1==index2:
				continue
			otherRow = row2
			# Choose comparison metric (distance/similarity)
			# modify desireHighValue False - distance, True - similarity
			if metricType == "euclidean":
				metric = euclideanDistance(selectedRow, otherRow)
				desireHighValue = False
			elif metricType == "manhattan":
				metric = manhattanDistance(selectedRow, otherRow)
				desireHighValue = False

			values.append((metric, index2))

		values.sort(key = itemgetter(0), reverse = desireHighValue)
		record = {'Transaction':index1}
		num = 1
		for (distance,ind) in values[0:k]:
			record[str(num)] = ind
			record[str(num) + "-prox"] = distance
			num = num+1
		rowList.append(record)

	df_ = pd.DataFrame(rowList)
	df_.to_csv("iris_" + metricType + "_out.csv")

	print
	print "Most common nearest neighbour:"
	print df_['1'].value_counts().head()

	# Proximity Distributions
	if k>=10:
		for proxVal in ['1-prox','5-prox','10-prox']:
			df_[[proxVal]].plot(kind='hist', title = "iris_" + metricType + "_Prox Value Distribution", figsize=(15, 10), legend=True, fontsize=12, xlim=(0,1))
			plt.savefig("iris_" + metricType + "_" + proxVal + " Value Distribution.png")

elif filename=='income_tr.csv':

	if (metricType!="gower") and (metricType!="binarizedManhattan"):
		print "Choose metric type from [gower/binarizedManhattan]"
		sys.exit()

	k = int(raw_input("Enter value of k: ") or "5") 
	print(k)

	columns = ['Transaction']
	for i in range(k):
		columns.append(str(i+1))
		columns.append(str(i+1) + "-prox")

	# Normalize data
	data.columns = data.columns.str.strip()
	cols_to_norm = ['capital_gain','capital_loss','hour_per_week']
	data[cols_to_norm] = data[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
	
	# Remove missing work data
	# Records with missing workclass and occupation = 28
	# data = data[~data.workclass.str.contains('^\s*\?$')] # removing this led to losing track of the index value of each record as per original dataset
	# will skip these rows while computing distances

	# After workclass based removal, 5 missing native_country values
	# Obs. for large number of records native_country = United States
	data.loc[data["native_country"].str.contains('^\s*\?$'),"native_country"] = "United-States"

	# Education = education_cat (redundancy)
	# ID = irrelevant
	data = data.drop(['education','ID','class'],1)

	numCols = []
	stringCols = []

	data.columns = data.columns.str.strip()
	for col in data.columns:
	    if(data[col].dtype == np.float64 or data[col].dtype == np.int64):
	    	numCols.append(col)
	    else:
	    	stringCols.append(col)

	# Normalize numeric ratio data and ordinal
	cols_to_norm = numCols
	data[cols_to_norm] = data[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

	if metricType=='binarizedManhattan':
		# Convert nominal attributes to dummy binary variables
		data_new = data[numCols]
		for col in stringCols:
			data_new = pd.concat([data_new,pd.get_dummies(data[col], prefix=col)], axis = 1)
		data = data_new
		# data  = data_new.drop(['workclass_ ?','occupation_ ?'],1)
		# Will ignore the rows that have 1 in these columns

	# Creating Dataframe with Proximity measures
	rowList = []
	for (index1,row1) in data.iterrows():
		if ('workclass' in data.columns and re.match('^\s*\?$',row1['workclass'])) or ('workclass_ ?' in data.columns and row1['workclass_ ?']==1):
			continue
		values = []
		selectedRow = row1
		for (index2,row2) in data.iterrows():
			if index1==index2 or ('workclass' in data.columns and re.match('^\s*\?$',row1['workclass'])) or ('workclass_ ?' in data.columns and row1['workclass_ ?']==1):
				continue
			otherRow = row2
			# Choose comparison metric (distance/similarity)
			# modify desireHighValue False - distance, True - similarity
			if metricType == "gower":
				metric = gowerSimilarity(selectedRow, otherRow)
				desireHighValue = True
			elif metricType == "binarizedManhattan":
				metric = manhattanDistance(selectedRow, otherRow)
				desireHighValue = False

			values.append((metric, index2))

		values.sort(key = itemgetter(0), reverse = desireHighValue)
		record = {'Transaction':index1}
		num = 1
		for (distance,ind) in values[0:k]:
			record[str(num)] = ind
			record[str(num) + "-prox"] = distance
			num = num+1
		rowList.append(record)

	df_ = pd.DataFrame(rowList)
	df_.to_csv("income_" + metricType + "_out.csv")

	print
	print "Most common nearest neighbour:"
	print df_['1'].value_counts().head()

	# Proximity Distributions
	if k>=10:
		for proxVal in ['1-prox','5-prox','10-prox']:
			df_[[proxVal]].plot(kind='hist', title = "income_" + metricType + "_Prox Value Distribution", figsize=(15, 10), legend=True, fontsize=12)
			plt.savefig("income_" + metricType + "_" + proxVal + " Value Distribution.png")


else:
	print "Invalid filename"
	sys.exit()



