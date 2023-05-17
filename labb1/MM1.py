import heapq
import random
import matplotlib.pyplot as plt
import numpy as np
from queue import Queue

# import matplotlib as mpl
# mpl.use('tkagg')
# import matplotlib.pyplot as plt

signalList = []
def send(signalType, evTime, destination, info):
    heapq.heappush(signalList, (evTime, signalType, destination, info))

GENERATE = 1
ARRIVAL = 2
MEASUREMENT = 3
DEPARTURE = 4

simTime = 0.0
stopTime = 10000.0


class larger():
    def __gt__(self, other):
        return False

class generator(larger):
    def __init__(self, sendTo,lmbda):
        self.sendTo = sendTo
        self.lmbda = lmbda
        self.arrivalTimes = []
    def arrivalTime(self):
        return simTime + random.expovariate(self.lmbda)
    def treatSignal(self, x, info):
        if x == GENERATE:
            send(ARRIVAL, simTime, self.sendTo, simTime)  #Send new cusomer to queue
            send(GENERATE, self.arrivalTime(), self, [])  #Schedule next arrival
            self.arrivalTimes.append(simTime)


class queue(larger):
    def __init__(self, mu, sendTo):
        self.numberInQueue = 0
        self.sumMeasurements = 0
        self.numberOfMeasurements = 0
        self.measuredValues = []
        self.buffer = Queue(maxsize=0) # ÄNDRA EJ!
        self.mu = mu 
        self.sendTo = sendTo
        self.numberBlocked = 0
        self.numberServed = 0
    def serviceTime(self):
        # return simTime + 0.1 - Deterministic
        # return simTime + np.random.uniform(low=0.0, high=0.2) - uniform distribution
        return simTime + random.expovariate(self.mu) #- exponential distribution
        
        # alpha = 0.25 - hyperexponential distribution
        # if random.uniform(0, 1) > alpha:
        #     return simTime + random.expovariate(1/0.13)
        # else:
        #     return simTime + random.expovariate(100)
            
        
    def treatSignal(self, x, info):
        if x == ARRIVAL:
            if self.buffer.qsize() < 6:
                q.numberServed = q.numberServed + 1
                if self.numberInQueue == 0:
                    send(DEPARTURE,self.serviceTime() , self, []) #Schedule  a departure for the arrival customer if queue is empty
                self.numberInQueue = self.numberInQueue + 1
                self.buffer.put(info)
            else:
                q.numberBlocked = q.numberBlocked + 1
        elif x == DEPARTURE:
            self.numberInQueue = self.numberInQueue - 1
            if self.numberInQueue > 0:
                send(DEPARTURE, self.serviceTime(), self, [])  # Schedule  a departure for next customer in queue
            send(ARRIVAL, simTime, self.sendTo, self.buffer.get())
        elif x == MEASUREMENT:
            self.measuredValues.append(self.numberInQueue)
            self.sumMeasurements = self.sumMeasurements + self.numberInQueue
            self.numberOfMeasurements = self.numberOfMeasurements + 1
            send(MEASUREMENT, simTime + random.expovariate(1), self, [])



class sink(larger):
    def __init__(self):
        self.numberArrived = 0
        self.departureTimes = []
        self.totalTime = 0
        self.T = []
    def treatSignal(self, x, info):
        self.numberArrived = self.numberArrived + 1
        self.departureTimes.append(info)
        self.totalTime = self.totalTime + simTime - info
        self.T.append(simTime - info)

          
  ###################################################
  #
  # Add code to create a queuing system  here
  #
  ###################################################
  
s = sink()
q = queue(10, s)
gen = generator(q, 7)

send(GENERATE, 0, gen, [])
send(MEASUREMENT, 10.0, q, [])

while simTime < stopTime:
    [simTime, signalType, dest, info] = heapq.heappop(signalList)
    dest.treatSignal(signalType, info)


  ###################################################
  #
  # Add code to print final result
  #
  ###################################################
  
a = list(range(0,7)) 
# plt.plot(q.measuredValues, a)
# plt.hist(q.measuredValues,bins=a, alpha=0.2)
# plt.hist(q2.measuredValues, bins=a, alpha=0.2)
# plt.plot(q.measuredValues)
# plt.plot(q2.measuredValues)
# lambda1 = float(input('Arrival rate (lambda): '))
# mu = float(input('Service rate (mu): '))
# maxK = int(input('Maximum k value to plot: '))

# k =  np.array([i for i in range(0,maxK)])
# rho = lambda1/mu
# pk = pow(rho,k)*(1-rho)

# plt.plot(k,pk,'-') 
# plt.grid(True)
# plt.xlabel('Number of packets in system, k')
# plt.ylabel('P(k)')
# plt.plot(q.measuredValues)
# plt.plot(q2.measuredValues)
# plt.plot(q3.measuredValues)
# plt.plot(q4.measuredValues)
# plt.plot(q5.measuredValues)
# plt.plot(q6.measuredValues)

# plt.plot(s.T[2:101],s.T[1:100],'*')
# plt.show()

# plt.plot(s3.T[2:101],s3.T[1:100],'*')


# plt.show()

# print('Probability of blocking: ', q.numberBlocked/(q.numberServed + q.numberBlocked))

## Uppgift 4.1 and beyond


plt.hist(q.measuredValues,density=True, bins=a)
plt.show()
print('Mean number in queue: ', np.mean(q.measuredValues))
print('Mean time in queue: ', np.mean(s.T))
print('Probability of blocking: ', q.numberBlocked/(q.numberServed + q.numberBlocked))
