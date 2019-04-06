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
samples=np.array(m)
mean=np.mean(samples)
var=np.var(samples)
std=np.sqrt(var)
print("This is the standard deviation:", std)
p=np.linspace(min(samples), max(samples),12)
print("Excess kurtosis of normal distribution ( should be 0):{}".format(stats.kurtosis(p)))
print("Skewness of normal distribution ( should be 0):{}".format(stats.skew(p)))
p_pdf=stats.norm.pdf(0.25,mean,std)
print("y_pdf",p_pdf)
p_cdf=stats.norm.cdf(0.25,mean,std)
print("y_cdf",p_cdf)
p_ppf=stats.norm.ppf(0.25,mean,std)
print("y_ppf",p_ppf)
p_skew_pdf=stats.skewnorm.pdf(p,*stats.skewnorm.fit(samples))
fit = stats.norm.pdf(samples, mean, std)
#this is a fitting indeed
pl.plot(samples,fit,'-o')
pl.hist(samples,normed=True)      #use this to draw histogram of 
pl.show()    