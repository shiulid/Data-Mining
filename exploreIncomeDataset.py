import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np
from pandas.tools.plotting import scatter_matrix
import matplotlib as mpl
from decimal import Decimal

p = '1'


mpl.rcParams["axes.color_cycle"] = ['#0000ff','#00ff00','#ff0000','#000000','#ffff00', 
'#551a8b','#ffa500','#add8e6','#f0dc82','#d3d3d3',
'#ad3a63','#e66721','#8a2be2','#ffcc00','#85274e',
'#adff2f','#791b19','#556b2f']
filename = "income_tr.csv"

"""
Nominal attributes with more than 2 values can 
be transformed to set of dummy binary variables
Hamming distance = length of different digits
http://people.revoledu.com/kardi/tutorial/Similarity/NominalVariables.html


Attributes:
age: Ratio
workclass: Nominal 		||
fnlwgt: (sampling weight) 
education: Ordinal :: education_cat
1< Preschool 
2< 1st-4th 
3< 5th-6th 
4< 7th-8th 
5< 9th 
6< 10th 
7< 11th 
8< 12th 
9< HS-grad 
10< Some-college
11< Assoc-voc 
12< Assoc-acdm 
13< Bachelors 
14< Masters
15< Prof-school
16< Doctorate.

education_cat: Ordinal
marital_status: Nominal
occupation: Nominal 		||
relationship: Nominal
race: Nominal
gender: Nominal binary attribute
capital_gain:
capital_loss:
hour_per_week:
native_country: Nominal
class:

"""

try:
	data = pd.read_csv(filename)
except IOError:
	print("Couldn't open file " + filename + "!")
	sys.exit()

numCols = []
stringCols = []
for col in data.columns:
    if(data[col].dtype == np.float64 or data[col].dtype == np.int64):
    	numCols.append(col)
    else:
    	stringCols.append(col)

if p == '1':
	print(data.describe())

	for label,group in data.groupby('class'):
		group.hist(xrot = 45, layout = (2,4))
		plt.title(label)

	data.hist()

	factor = data['class']

	classes = list(set(factor))

	palette = ['#e41a1c', '#377eb8', '#4eae4b', 
		'#994fa1', '#ff8101', '#fdfc33',
		'#a8572c', '#f482be', '#999999']

	color_map = dict(zip(classes,palette))

	print color_map

	colors = factor.apply(lambda group: color_map[group])
	axarr = scatter_matrix(data,figsize=(10,10),marker='o',c=colors,diagonal=None)
	plt.show()

	# Check for missing data "?"
	
	"""
	Missing & Unique values per column
	"""
	for col in stringCols:
		data[col] = data[col].str.strip()
		print "Attribute-",col,":"
		print "Missing:",np.sum(data[col].str.contains('^\?$'))
		print "Unique Values:",len(data[col].unique())
		print

	"""
	Missing values per row
	"""
	count0 = 0
	count1 = 0
	count2 = 0
	count3 = 0
	count4 = 0
	for (index1,row1) in data.iterrows():
		
		missing = np.sum(row1.str.contains('^\?$'))
		if missing == 1:
			count1 += 1
		elif missing == 2:
			count2 += 1
		elif missing == 0:
			count0 += 1
		elif missing == 3:
			count3 += 1
		else:
			count4 += 1
	li = [(0,count0), (1,count1), (2,count2), (3,count3), (4,count4)]

	dictionary = plt.figure()

	D = {0:count0, 1: count1, 2:count2,3:count3,4:count4}

	plt.bar(range(len(D)), D.values(), align='center')
	plt.xticks(range(len(D)), D.keys())
	plt.title("No. of rows with N number of missing attributes")
	plt.show()


	"""
	NUMERICAL
	"""
	"""
	Unique
	"""
	dictionary = plt.figure()

	D = {}

	numCols.remove('fnlwgt')
	numCols.remove('ID')

	for col in numCols:

		D = data[col].value_counts(normalize="True").to_dict()
		if col=="capital_loss" or col=="capital_gain":
			D.pop(0, None)

		plt.bar(range(len(D)), D.values(), align='center')
		plt.xticks(range(len(D)), D.keys(),rotation=90)

		plt.title(col)
		plt.show()

#else:

	"""
	CATEGORICAL
	"""
	"""
	Unique
	"""

	dictionary = plt.figure()

	D = {}

	for col in stringCols:

		D = data[col].value_counts(normalize="True").to_dict()

		print col
		print "{:<20} {:<15}".format('Value','Percentage')
		for k, v in D.iteritems():
			print "{:<20} {:<15}".format(k, Decimal(v*100).quantize(Decimal('0.01')))
		print

		plt.bar(range(len(D)), D.values(), align='center')
		plt.xticks(range(len(D)), D.keys(),rotation=90)

		plt.title(col)
		plt.show()

	""" 
	Classwise distribution (for Nominal) 
	"""
	class_mapping = {
	           '<=50K': 0,
	           '>50K': 1}

	data['class'] = data['class'].map(class_mapping)


	for col in stringCols:
		rlist = []
		for label, df in data.groupby(col):
			count0 = np.sum(df['class']==0)
			count1 = np.sum(df['class']==1)
			rlist.append({'value':label,'<=50K':count0,'>50K':count1})

		df_ = pd.DataFrame(rlist)

		df_.plot.bar(stacked="True",title=col)
		plt.xticks(range(len(df_['value'])), df_['value'],rotation=45)
		plt.show()

	"""
	Boxplot for Numeric
	"""

	#data.groupby('class').plot.box()

	for col in numCols:	
		data[['class',col]].boxplot(by='class')
		plt.show()


	"""
	Relationship with Age
	"""

	for col in ['workclass','education','marital_status','occupation','relationship','race','gender','native_country']:

		bins = np.array([0,10,20,30,40,50,60,70,80,90,100])
		d = np.digitize(data['age'],bins)
		df1 = pd.DataFrame({'AgeBin':d, col:data[col]})
		rlist = []
		for label,group in df1.groupby('AgeBin'):
			D = {}
			for val in data[col].unique():
				D[val] = np.sum(group[col]==val)
			rlist.append(D)
		df_ = pd.DataFrame(rlist)
		df_.plot.bar(stacked=True)
		plt.xticks(range(len(bins[2:])), bins[2:],rotation=45)
		plt.title(col)
		plt.xlabel('Age Bins')
		plt.show()

	# 7:80-89
	# 0:10-19

	"""
	classwise hours_per_week
	"""
	for col in ['native_country']:

		bins = np.array([0,10,20,30,40,50,60,70,80,90,100])
		d = np.digitize(data['hour_per_week'],bins)
		df1 = pd.DataFrame({'HoursBin':d, col:data[col]})
		rlist = []
		for label,group in df1.groupby('HoursBin'):
			D = {}
			for val in data[col].unique():
				D[val] = np.sum(group[col]==val)
			rlist.append(D)
		df_ = pd.DataFrame(rlist)
		df_.plot.bar(stacked=True)
		plt.xticks(range(len(bins[2:])), bins[2:],rotation=45)
		plt.title(col)
		plt.xlabel('hours_per_week bins')
		plt.show()

	plt.show()