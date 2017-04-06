import sys
import pandas as pd 
from pandas.util.testing import assert_frame_equal
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans

def euclideanDistance(rec1, rec2):
	return math.sqrt(sum(math.pow(val1 - val2,2) for (val1,val2) in zip(rec1,rec2)))

if len(sys.argv) < 2:
	print "Enter: <dataset:easy|hard|wine>"
	sys.exit()

arg = sys.argv[1]
if arg == "easy":
	filename = "TwoDimEasy.csv"
elif arg == "hard":
	filename = "TwoDimHard.csv"
elif arg == "wine":
	filename = "wine.csv"
else:
	print "Enter: <dataset:easy|hard|wine>"
	sys.exit()


df = pd.read_csv(filename)

#print df.head()

if arg in {'easy','hard'}:
	numCols = ['X.1', 'X.2']
elif arg == 'wine':
	numCols = df.drop(['ID','class','quality'], axis=1).columns

df[numCols] = df[numCols].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

k = int(raw_input("Enter value of k: ") or "2") 

sample = df.sample(n=k)

newCentroids = sample[numCols]

## sklearn KMeans
#kmeans = KMeans(n_clusters=k).fit(df[numCols])
#df['Predicted Cluster'] = kmeans.labels_
# kmeans = KMeans(n_clusters=k).fit(df[numCols])
# df['Predicted Cluster'] = kmeans.labels_

# sample = df.sample(n=k)
# newCentroids = sample[numCols]

# index = newCentroids.index
# newCentroids.ix[index[0],'X.1'] = kmeans.cluster_centers_[0][0]
# newCentroids.ix[index[0],'X.2'] = kmeans.cluster_centers_[0][1]
# newCentroids.ix[index[1],'X.1'] = kmeans.cluster_centers_[1][0]
# newCentroids.ix[index[1],'X.2'] = kmeans.cluster_centers_[1][1]


while True:
	#print newCentroids
	#print
	prevCentroids = newCentroids.copy()
	for (index, row) in df.iterrows():
		distance = [euclideanDistance(row[numCols], rowC[numCols]) for (_,rowC) in prevCentroids.iterrows()]
		df.ix[index,'Predicted Cluster'] = int(distance.index(min(distance)))
	newCentroids = df.groupby('Predicted Cluster')[numCols].mean()

	if prevCentroids.equals(newCentroids):
		break

if arg in {'easy','hard'}:
	cluster = 'cluster'
else:
	cluster = 'class'

c = df[numCols].mean()
## SSE & SSB

# True Cluster SSE & SSB
print 
print "True Clusters SSE, SSB -"
trueClusters = df.groupby(cluster)[numCols]
totSSE = 0
totSSB = 0
for ((name,group), i) in zip(trueClusters, range(k)):
	centroid = group[numCols].mean()

	# SSE
	distance = [euclideanDistance(row[numCols], centroid) for (_,row) in group.iterrows()]
	sse = sum([math.pow(d,2) for d in distance])
	totSSE = totSSE + sse

	# SSB
	ssbTerm = group.shape[0] * math.pow(euclideanDistance(c,centroid),2)
	totSSB = totSSB + ssbTerm

	print "Cluster: ", name, "\tSSE: ", sse
print "Overall SSE: ", totSSE
print "Overall SSB:", totSSB
print "Overall Measure: ", totSSB + totSSE

# Predicted Cluster SSE & SSB
print 
print "Predicted Clusters SSE, SSB -"
newClusters = df.groupby('Predicted Cluster')[numCols]

totSSE = 0
totSSB = 0
for ((name,group), i) in zip(newClusters, range(k)):
	centroid = newCentroids.iloc[i]

	# SSE
	distance = [euclideanDistance(row[numCols], centroid) for (_,row) in group.iterrows()]
	sse = sum([math.pow(d,2) for d in distance])
	totSSE = totSSE + sse

	# SSB
	ssbTerm = group.shape[0] * math.pow(euclideanDistance(c,centroid),2)
	totSSB = totSSB + ssbTerm

	print "Cluster: ", name, "\tSSE: ", sse
print "Overall SSE: ", totSSE
print "Overall SSB:", totSSB
print "Overall Measure: ", totSSB + totSSE



# Cross tabulation matrix
print 

df[['ID',cluster, 'Predicted Cluster']].to_csv("output_"+filename+".csv", index = False)

print df.groupby('Predicted Cluster')[cluster].value_counts()

if arg=='wine':
	test = df.groupby(['Predicted Cluster',cluster])[cluster].count().unstack(cluster).fillna(0)
	test.plot(kind='bar', stacked=True)
	test = df.groupby(['Predicted Cluster','quality'])['quality'].count().unstack('quality').fillna(0)
	test.plot(kind='bar', stacked=True)

plt.show()

if arg in {'easy','hard'}:
	plt.subplot(121)
	plt.scatter(df['X.1'], df['X.2'], c=df['Predicted Cluster'])
	plt.title('Dataset: '+filename+ "[Predicted]")
	plt.gca().set_aspect('equal', adjustable='box')
	plt.subplot(122)
	plt.scatter(df['X.1'], df['X.2'], c=df['cluster'])
	plt.title('Dataset: '+filename+" [Actual]")
	plt.gca().set_aspect('equal', adjustable='box')
	plt.show()

