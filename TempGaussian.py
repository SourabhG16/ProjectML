import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pylab as pl
from collections import Counter
#Read the txt file
data=open("weather.txt",'r')
x=data.read()
print (x)
print (type(x))
#maps all string-numbers to integer as separated by commas
n=list(map(float,x.split(',')))
print (n)
#Testing the claim that a student with 350 ms is faster than me
under350=[x for x in n if x<=350]
#Count the number of occurrences of a number
print (under350)
print (Counter(under350))
print (Counter(n))
print ("This is the minimum number:", min(n))
print ("This is the maximum number:", max(n))
print ("This is the mean number:", sum(n)/len(n))
# Convert list to numpy array
samples=np.array(n)
mean=np.mean(samples)
var=np.var(samples)
std=np.sqrt(var)
print("This is the standard deviation:", std)
x=np.linspace(min(samples), max(samples),12)
print("Excess kurtosis of normal distribution ( should be 0):{}".format(stats.kurtosis(x)))
print("Skewness of normal distribution ( should be 0):{}".format(stats.skew(x)))
y_pdf=stats.norm.pdf(25,mean,std)
print("y_pdf",y_pdf)
y_cdf=stats.norm.cdf(25,mean,std)
print("y_cdf",y_cdf)
y_ppf=stats.norm.ppf(25,mean,std)
print("y_ppf",y_ppf)
y_skew_pdf=stats.skewnorm.pdf(x,*stats.skewnorm.fit(samples))
fit = stats.norm.pdf(samples, mean, std)
#this is a fitting indeed
pl.plot(samples,fit,'-o')
pl.hist(samples,normed=True)      #use this to draw histogram of 
pl.show() 
  
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
a = np.array([58,48,38,30,50,72,82,83,80,69,65,64])
c = multivariate_normal.cdf(22, mean=62.417, cov=8.864); 
print(c)
b=np.	array([1,2,2,4,7,7,6,5,3,2,2,1])
plt.plot(a,b)
#plt.show()

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pylab as pl
from collections import Counter
#Read the txt file
data=open("my_reaction_time_50samples.txt",'r')
p=data.read()
print (p)
print (type(p))
#maps all string-numbers to integer as separated by commas
m=list(map(float,p.split(',')))
print (m)
#Testing the claim that a student with 350 ms is faster than me
under350=[p for p in m if p<=350]
#Count the number of occurrences of a number
print (under350)
print (Counter(under350))
print (Counter(m))
print ("This is the minimum number:", min(m))
print ("This is the maximum number:", max(m))
print ("This is the mean number:", sum(m)/len(m))
# Convert list to numpy array
samples1=np.array(m)
mean1=np.mean(samples)
var1=np.var(samples)
std1=np.sqrt(var)
print("This is the standard deviation:", std1)
p=np.linspace(min(samples1), max(samples1),12)
print("Excess kurtosis of normal distribution ( should be 0):{}".format(stats.kurtosis(p)))
print("Skewness of normal distribution ( should be 0):{}".format(stats.skew(p)))
p_pdf=stats.norm.pdf(0.25,mean1,std1)
print("y_pdf",p_pdf)
p_cdf=stats.norm.cdf(0.25,mean1,std1)
print("y_cdf",p_cdf)
p_ppf=stats.norm.ppf(0.25,mean1,std1)
print("y_ppf",p_ppf)
p_skew_pdf=stats.skewnorm.pdf(p,*stats.skewnorm.fit(samples))
fit1 = stats.norm.pdf(samples1, mean1, std1)
#this is a fitting indeed
pl.plot(samples1,fit1,'-o')
pl.hist(samples1,normed=True)      #use this to draw histogram of 
pl.show()    
print((y_cdf+p_cdf+c))