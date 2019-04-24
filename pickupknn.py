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
	distances = []
	for index,row in data.iterrows():
		euclidean_dist=np.sqrt(((row['Day']-predict[0])**2)+(row['Probability']-predict[1])**2)
		distances.append([euclidean_dist,row['demand']])
	lab1=distances[:k]
	print("distances:",lab1)
	lab2=np.array(lab1)
	result=round(np.mean(lab2[:,1]))
	print(round(np.mean(lab2[:,1])))
	for index,row in data.iterrows():
		plt.scatter(row['Probability'],row['demand'], s=100, color='r') 
	plt.scatter(predict[1],result,marker='x',label='pickup')
	plt.legend(loc=4)
	plt.show()
	row = [predict[0],predict[1],result]
	with open('Hisnew.csv', 'a',newline='') as csvFile:
		writer=csv.writer(csvFile)
		writer.writerow(row) 
	csvFile.close()
dataset=pd.read_csv('Hisnew.csv')
new_point=[230,1.060973637]
k_nearest_neighbor(dataset,new_point,k=11)




