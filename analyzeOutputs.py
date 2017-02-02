import pandas as pd 
import sys
from collections import Counter

if len(sys.argv)<3:
	print "Input Format: <actual csv> <outputcsv>"
	sys.exit()

inputFile = sys.argv[1]
outputFile = sys.argv[2]

try:
	dataIn = pd.read_csv(inputFile)
	dataOut = pd.read_csv(outputFile)
except IOError:
	print IOError
	sys.exit()

dataIn.columns = dataIn.columns.str.strip()
dataOut.columns = dataOut.columns.str.strip()

classes = list(set(dataIn['class']))

n = len(dataOut.columns)/2 - 1

if n%2==0:
	n = n-1
while n>=1:

	incorrectCount = 0
	for (index,row) in dataOut.iterrows():
		actualClass = dataIn.iloc[int(row['Transaction'])]['class']
		votes = []
		for i in range(n):
			votes.append(dataIn.iloc[int(row[i+1])]['class'])
		data = Counter(votes)
		if data.most_common(1)[0][0] != actualClass:
			incorrectCount += 1

	accuracy = 1 - (float(incorrectCount)/dataIn.shape[0])
	print "n =", n, ", accuracy = ", accuracy

	n -= 2