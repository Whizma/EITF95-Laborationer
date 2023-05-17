import random
import matplotlib.pyplot as plt
import numpy as np



#import matplotlib as mpl
#mpl.use('tkagg')
#import matplotlib.pyplot as plt




lambda1 = float(input('Arrival rate (lambda): '))
mu = float(input('Service rate (mu): '))
maxK = int(input('Maximum k value to plot: '))


k =  np.array([i for i in range(0,maxK)])
rho = lambda1/mu
pk = pow(rho,k)*(1-rho)




plt.plot(k,pk,'-') 
plt.grid(True)
plt.xlabel('Number of packets in system, k')
plt.ylabel('P(k)')
plt.show()






