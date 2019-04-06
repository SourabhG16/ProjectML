import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pylab as pl
from collections import Counter
#Read the txt file
data=open("weather.txt",'r')
x=data.read()
#print(x)
#print(type(x))
#maps all string-numbers to integer as separated by commas
n=list(map(float,x.split(',')))
#print(n)
#Testing the claim that a student with 350 ms is faster than me
under350=[x for x in n if x<=350]
#Count the number of occurrences of a number
#print(under350)
#print(Counter(under350))
#print(Counter(n))
#print("This is the minimum number:", min(n))
#print("This is the maximum number:", max(n))
#print("This is the mean number:", sum(n)/len(n))
# Convert list to numpy array
samples=np.array(n)
mean=np.mean(samples)
var=np.var(samples)
std=np.sqrt(var)
#print("This is the standard deviation:", std)
x=np.linspace(min(samples), max(samples),12)
#print("Excess kurtosis of normal distribution ( should be 0):{}".format(stats.kurtosis(x)))
#print("Skewness of normal distribution ( should be 0):{}".format(stats.skew(x)))
data1=open("WeatherTraining.txt",'r')
x1=data1.read()
x12=list(map(float,x1.split(',')))
#print(x1)
xiterate= np.array(x12)
xiterate= xiterate.reshape(5,73)
#print("training is this")
y_cdf=np.array([[0 ,0,0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0 ,0,0,0 ,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0 ,0,0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0 ,0,0,0 ,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0 ,0,0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0 ,0,0,0 ,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0 ,0,0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0 ,0,0,0 ,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0 ,0,0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0 ,0,0,0 ,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],dtype='f')
for i in range(0,5):
	for j in range(0,73):
		y_cdf[i][j]=stats.norm.cdf(j,mean,std)
print("y_cdf",y_cdf)
y_skew_pdf=stats.skewnorm.pdf(x,*stats.skewnorm.fit(samples))
fit = stats.norm.pdf(samples, mean, std)
#this is a fitting indeed
pl.plot(samples,fit,'-o')
pl.hist(samples,normed=True)
#use this to draw histogram of 
pl.show() 
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
humid=[57,53,56,52,62,61,56,54,55,58,62,58,61,60,60,57,57,54,57,57,58,61,60,57,61,57,55,58,62,59,59,48,51,48,53,47,49,46,49,44,46,44,48,54,53,45,54,44,48,48,52,49,48,44,47,46,54,52,49,37,32,41,36,39,39,37,39,35,33,36,36,34,41,32,37,37,39,33,32,37,38,42,32,41,39,42,35,33,38,32,24,24,25,24,25,28,30,29,29,27,32,34,30,27,27,30,27,27,30,30,31,31,33,24,30,33,27,28,33,27,29,29,47,44,53,47,47,49,50,52,52,46,53,51,45,51,43,52,46,52,47,45,47,50,53,46,48,46,47,47,52,45,69,69,72,65,74,67,72,72,70,73,65,65,66,70,67,69,70,72,69,70,73,65,70,66,66,74,74,69,74,68,76,77,74,79,80,75,78,77,75,76,82,81,75,78,79,74,83,80,81,77,81,83,77,78,83,75,76,83,75,76,87,78,80,83,81,78,80,86,78,83,79,80,78,82,82,86,80,82,86,83,84,77,81,87,77,87,84,79,79,79,79,81,75,80,76,77,83,74,74,79,76,81,83,75,74,79,79,79,82,77,78,73,74,83,74,73,79,82,73,83,82,64,69,59,68,60,65,59,65,59,62,65,62,69,69,59,69,59,65,64,65,61,67,66,68,64,59,66,66,68,64,55,61,56,54,54,53,62,62,59,62,54,56,58,57,57,57,55,55,57,55,53,63,53,56,63,62,59,58,60,54,53,55,61,56,54,53,62,62,59,62,54,56,58,57,57,57,55,55,57,55,53,63,53,56,63,62,59,58,60,54,53,58]
a = np.array([56,46,36,36,48,70,79,82,78,64,58,58])
for i in humid:
	c = multivariate_normal.cdf(i, mean=62.417, cov=	8.864); 
	print(c)
b=np.	array([1,2,2,4,7,7,6,5,3,2,2,1])
plt.plot(a,b)
print(len(humid))
#plt.show()

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pylab as pl
from collections import Counter
#Read the txt file
data=open("WtTypes.txt",'r')
p=data.read()
print (p)
print (type(p))
#maps all string-numbers to integer as separated by commas
m=list(map(float,p.split(',')))
print (m)
#Testing the claim that a student with 350 ms is faster than me
under350=[p for p in m if p<=350]
#Count the number of occurrences of a number
#print (under350)
#print (Counter(under350))
#print (Counter(m))
#print ("This is the minimum number:", min(m))
#print ("This is the maximum number:", max(m))
#print ("This is the mean number:", sum(m)/len(m))
#Convert list to numpy array
samples1=np.array(m)
mean1=np.mean(samples)
var1=np.var(samples)
std1=np.sqrt(var)
#print("This is the standard deviation:", std1)
p=np.linspace(min(samples1), max(samples1),12)
#print("Excess kurtosis of normal distribution ( should be 0):{}".format(stats.kurtosis(p)))
#print("Skewness of normal distribution ( should be 0):{}".format(stats.skew(p)))
data1=open("WtTypesTraining.txt",'r')
y1=data1.read()
y12=list(map(float,y1.split(',')))
#print(y1)
yiterate= np.array(y12)
yiterate= yiterate.reshape(5,73)
print("training is this")
#np.zeros((5, 73))
p_cdf=np.array([[0 ,0,0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0 ,0,0,0 ,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0 ,0,0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0 ,0,0,0 ,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0 ,0,0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0 ,0,0,0 ,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0 ,0,0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0 ,0,0,0 ,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0 ,0,0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0 ,0,0,0 ,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],dtype='f')
print(a)
for i in range(0,5):
	for j in range(0,73):
		p_cdf[i][j]=stats.norm.cdf(j,mean1,std1)
		
p_skew_pdf=stats.skewnorm.pdf(p,*stats.skewnorm.fit(samples))
fit1 = stats.norm.pdf(samples1, mean1, std1)
#this is a fitting indeed
pl.plot(samples1,fit1,'-o')
pl.hist(samples1,normed=True)     
#use this to draw histogram of 
pl.show()

import random 
d=random.uniform(0.5,1.0)
e=random.uniform(0.5,1.0)
f=random.uniform(0.5,1.0)
similarity=np.array([[0 ,0,0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0 ,0,0,0 ,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0 ,0,0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0 ,0,0,0 ,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0 ,0,0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0 ,0,0,0 ,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0 ,0,0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0 ,0,0,0 ,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0 ,0,0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0 ,0,0,0 ,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],dtype='f')
print("similarity")
for i in range(0,5):
	for j in range(0,73):
		similarity[i][j]=d*y_cdf[i][j]+e*p_cdf[i][j]+c*f
		print(similarity[i][j])


import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd  
# predd = pd.read_csv('His.csv',error_bad_lines=False) 
predd = pd.read_csv('His.csv') 
predd=pd.DataFrame(predd)
predd.dropna()
print(predd)
names = ['Day','Pickup','Prob']
# Read dataset to pandas dataframe
X = predd.iloc[:,0:2].values  
y = predd.iloc[:,-1].values  
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)  
from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=5)  
classifier.fit(X_train, y_train)  
y_pred = classifier.predict(X_test) 
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))  
error = []
for i in range(1, 40):  
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i !=y_test))
plt.figure(figsize=(12, 6))  
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')  
plt.xlabel('K Value')  
plt.ylabel('Mean Error') 
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))

