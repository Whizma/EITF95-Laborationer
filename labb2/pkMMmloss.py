import numpy as np


#Calculates the steady state probabilities an M/M/m* loss system
#without queue,  Erlangfallet. The parameters required are offered
#traffic (rho) and
#number of servers (m).

rho = float(input('Traffic: '))
m = int(input('Number of servers: '))

def pk(k):
    """This is a recursive function """
    if k == 0:
        return 1
    else:
        return (pk(k-1)*rho/(k))



pr = np.array([pk(i) for i in range(m+1)])
p = pr/np.sum(pr)
pb = p[-1]


print("The blocking probabilty is ",pb)
  
