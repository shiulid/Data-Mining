import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np
from pandas.tools.plotting import scatter_matrix
import matplotlib.colors

filename = "Iris.csv"

data = pd.read_csv(filename)

print data.describe()

data.columns = data.columns.str.strip()

#Normalize data
#cols_to_norm = ['sepal_length','sepal_width','petal_length','petal_width']
#data[cols_to_norm] = data[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

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

print data.head()

data['sepal l:w'] = data['sepal_length'] / data['sepal_width']
data['petal l:w'] = data['petal_length'] / data['petal_width']
print data.head()

fig, ax = plt.subplots()
colors = color_map
grouped = data.groupby('class')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='sepal l:w', y='petal l:w', label=key, color=colors[key])

plt.show()
