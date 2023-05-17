import random
import matplotlib.pyplot as plt
import numpy as np


#import matplotlib as mpl
#mpl.use('tkagg')
#import matplotlib.pyplot as plt




lambda1 = float(input('Arrival rate (lambda): '))
mu = float(input('Service rate (mu): '))
L= int(input('Buffer size: ')) + 1  # 1  server
maxK = int(input('Maximum k value to plot: '))


rho = lambda1/mu


k =  np.array([i for i in range(0,L+1)])

if lambda1 == mu:
    pk = np.array([1/(L+1) for i in range(0,L+1)])
else:
    pk = pow(rho,k)*(1-rho)/ (1 - pow(rho,L+1))


lambdaEff = lambda1 * (1-pk[-1])


print('Effective arrival rate is :', lambdaEff)


pkL = np.zeros(maxK-L-1)  # probability k > L+1
pk = np.concatenate((pk, pkL), axis=None) 
k =  np.array([i for i in range(0,maxK)])


plt.plot(k,pk,'*') 
plt.grid(True)
plt.xlabel('Number of packets in system, k')
plt.ylabel('P(k)')
plt.show()






