import numpy as np
import csv
import matplotlib.pyplot as plt  
import pandas as pd  
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter

def k_nearest_neighbor(data,predict,k=3):
	# if len(dataset)>k:
	# 	warnings.warn('K is greater then data size.!')
	distances = []
	for index,row in data.iterrows():
		# print(row['Day'],"  ",row['Probability'])
		euclidean_dist=np.sqrt(((row['Day']-predict[0])**2)+(row['Probability']-predict[1])**2)
		distances.append([euclidean_dist,row['demand']])
	lab1=distances[:k]
	print("distances:",lab1)
	lab2=np.array(lab1)
	result=round(np.mean(lab2[:,1]))
	print(round(np.mean(lab2[:,1])))
	# lab3=Counter(lab2[:,1])
	# lab=[i[1] for i in sorted(distances)[:k]]		#sort according to euclid_dist take first k
	# print("labbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
	# print(lab)
	# result= Counter(lab).most_common(1)[0][0]
	# print("result",result_data[result]);
	# mi=marker=r"$ {} $".format(result_data[result])	
	for index,row in data.iterrows():
		plt.scatter(row['Probability'],row['demand'], s=100, color='r') 
	plt.scatter(predict[1],result,marker='x',label='pickup')
	# plt.scatter(predict[0],predict[1]-0.3,marker=mi,color=result,s=1000)
	plt.legend(loc=4)
	plt.show()
	row = [predict[0],predict[1],result]
	with open('Hisnew.csv', 'a',newline='') as csvFile:
		writer=csv.writer(csvFile)
		writer.writerow(row) 
	csvFile.close()
dataset=pd.read_csv('Hisnew.csv')
# print(dataset)
new_point=[230,1.060973637]
k_nearest_neighbor(dataset,new_point,k=11)




