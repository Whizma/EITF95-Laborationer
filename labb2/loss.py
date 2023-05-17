
import heapq
import random
import numpy as np
from queue import Queue
import time

#import matplotlib as mpl
#mpl.use('tkagg')
import matplotlib.pyplot as plt


MaxM = max(0,int(input('Maximum value of M: ')))
serv = 1
beta = 0.2

while MaxM > 0:
     mu=1/serv
     M =  np.array([i for i in range(1,MaxM)])
     P=np.ones([len(M),4])
     
     P[:,1] = M  * beta  /mu
     P[:,2] =  (M-1) * beta * P[:,2]/ (2*mu); 
     P[:,3] =  (M-2) * beta * P[:,3]/ (3*mu); 

     P[P<0] = 0

     P0 = np.zeros(len(M))
     for i in range(len(M)):
         P0[i] = 1/np.sum(P[i,:])
     

     E = np.zeros(len(M))
     B = np.zeros(len(M))  #blocking probability
     
     E = P[:,3] * P0   #P3

     B= (M-3) * P[:,3] / (M *P[:,0] + (M-1) * P[:,1] + (M-2) * P[:,2] + (M-3) * P[:,3])


     plt.plot(M,E,'-',M,B,'x') 
    
     plt.grid(True)
     plt.xlabel('Number of customers, M')
     plt.ylabel('Loss probability')
     plt.show()

     MaxM=0
   

 
  



