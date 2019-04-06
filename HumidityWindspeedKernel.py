import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
a = np.array([58,48,38,30,50,72,82,83,80,69,65,64])
b = multivariate_normal.cdf(22, mean=62.417, cov=8.864); 
print(b)
b=np.	array([1,2,2,4,7,7,6,5,3,2,2,1])
plt.plot(a, b)
#plt.show()