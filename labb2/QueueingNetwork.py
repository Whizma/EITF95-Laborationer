import heapq
import random
import numpy as np
from queue import Queue
import time

import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt

signalList = []

def send(signalType, evTime, destination, info):
    heapq.heappush(signalList, (evTime, signalType, destination, info))

GENERATE = 1
ARRIVAL = 2
MEASUREMENT = 3
DEPARTURE = 4

simTime = 0.0
stopTime = 10000.0
bufferSize = 4
        
class larger():
    def __gt__(self, other):
        return False


class generator(larger):
    def __init__(self, sendTo,lambda1):
        self.sendTo = sendTo
        self.lambda1 = lambda1
        self.arrivalTimes = []
    def arrivalTime(self):
        return simTime + random.expovariate(self.lambda1)
        # return simTime + 1.0
    def treatSignal(self, x, info):
        if x == GENERATE:
            send(ARRIVAL, simTime, self.sendTo, simTime)
            send(GENERATE, self.arrivalTime(), self, [])
            self.arrivalTimes.append(simTime)


class queue(larger):
    def __init__(self, mu, L, sendTo1,sendTo2,alpha,s):
        self.numberInQueue = 0
        self.sumMeasurements = 0
        self.numberOfMeasurements = 0
        self.measuredValues = []
        self.measuredInQueue = []
        self.arrivalTimes = []
        self.serviceTimes = []
        self.buffer = Queue(maxsize=0)
        self.mu = mu 
        self.L = L
        self.alpha = alpha
        self.sendTo1 = sendTo1
        self.sendTo2 = sendTo2
        self.s = s

    def serviceTime(self):
        # EXPONENTIAL
        # serviceTime = random.expovariate(self.mu)
        # self.serviceTimes.append(serviceTime)
        # return simTime + serviceTime

        # # CONSTANT
        # serviceTime = 1/self.mu
        # self.serviceTimes.append(serviceTime)
        # return simTime + serviceTime
        
        # HYPEREXPONENTIAL
        hyperA = 1/1.5 # hyperexponential distribution
        if random.uniform(0, 1) < hyperA:
            serviceTime = random.expovariate(2*self.mu)
            self.serviceTimes.append(serviceTime)
        else:
            serviceTime = random.expovariate(self.mu/2)
            self.serviceTimes.append(serviceTime)
        return simTime + serviceTime
        
    def treatSignal(self, x, info):
        if x == ARRIVAL:
            if self.numberInQueue < self.L: 
                if self.numberInQueue == 0:
                    send(DEPARTURE, self.serviceTime(), self, []) #Schedule  a departure for the arrival customer if queue is empty
                self.numberInQueue = self.numberInQueue + 1
            self.buffer.put(info)
            self.arrivalTimes.append(simTime)

        elif x == DEPARTURE:
            self.numberInQueue = self.numberInQueue - 1
            if self.numberInQueue > 0:
                send(DEPARTURE,  self.serviceTime(), self, [])  # Schedule  a departure for next customer
            tid = self.buffer.get()
            if random.uniform(0, 1) <= self.alpha :
               if self.sendTo1:
                  send(ARRIVAL, simTime, self.sendTo1, tid)
               send(ARRIVAL, simTime, self.s, tid)
            else : # rand >= self.alpha:
               if self.sendTo2:
                  send(ARRIVAL, simTime, self.sendTo2, tid) 
               send(ARRIVAL, simTime, self.s, tid)


        elif x == MEASUREMENT:
            self.measuredValues.append(self.numberInQueue)
            if (self.numberInQueue == 0):
                self.measuredInQueue.append(0)
            else:
                self.measuredInQueue.append(self.numberInQueue - 1)
                
            self.measuredValues.append(self.numberInQueue)
            self.sumMeasurements = self.sumMeasurements + self.numberInQueue
            self.numberOfMeasurements = self.numberOfMeasurements + 1
            send(MEASUREMENT, simTime + random.expovariate(1), self, [])

            
class sink(larger):
    def __init__(self):
        self.numberArrived = 0
        self.departureTimes = []
        self.totalTime = 0
        self.T = [] # Total time since the customer arrived to queuing network
    def treatSignal(self, x, info):
        self.numberArrived = self.numberArrived + 1
        self.departureTimes.append(info)
        self.totalTime = self.totalTime + simTime - info
        self.T.append(simTime - info)






# Parameters are given values
[ lambda1,lambda2] = [7.5, 10]  #Arrival rate  
[mu1,mu2,mu3,mu4,mu5] = [10, 14, 22, 9, 11]     # Service rate
[L1,L2,L3,L4,L5] = [np.inf, np.inf, np.inf, np.inf, np.inf]    
# [L1,L2,L3,L4,L5] = [4, 10, 3, 20, 7] 
alpha = 0.4

# queuing system below

startTime = time.time()
s = sink()
s2 = sink()
q4 = queue(mu4, L4, None, None, 1, s2)
q5 = queue(mu5, L2, None, None, 1, s2)
q3 = queue(mu3, L3, q4, q5, alpha, s)
q1 = queue(mu1, L1, q3, q3, 1 ,s)
q2 = queue(mu2, L2, q3, q3, 1, s)
queues = [q1, q2, q3, q4, q5]
gen1 = generator(q1, lambda1)
gen2 = generator(q2, lambda2)
send(GENERATE, 0, gen1, [])
send(GENERATE, 0, gen2, [])

for i in queues:
    send(MEASUREMENT, 1, i, [])

while simTime < stopTime:
    [simTime, signalType, dest, info] = heapq.heappop(signalList)
    dest.treatSignal(signalType, info)


nq = len(queues)
for i in range(nq):
    print('In queueing system', i+1, ': ',
    sum(queues[i].measuredInQueue) / len(queues[i].measuredInQueue),
    '\n Service time:',
    sum(queues[i].serviceTimes)/len(queues[i].serviceTimes),
    '\n'
    )    

print('Mean time in queue: ', s.totalTime/s.numberArrived)

totalTime = time.time() - startTime
print('Elapsed time: ', totalTime)
print('Total time in system: ', np.mean(s2.T))
print('Sum of mean number in queue: ', np.sum(np.mean(q1.measuredInQueue) + np.mean(q2.measuredInQueue) + np.mean(q3.measuredInQueue) + np.mean(q4.measuredInQueue) + np.mean(q5.measuredInQueue)))
        


  ###################################################
  #
  # Add code to print final result
  #
  ###################################################
  
# Uncomment the code below to se histograms of the number of customers in all the
# queues in the queueing network
  
# ypoints1 = np.array(q1.measuredValues)
# ypoints2 = np.array(q2.measuredValues)
# ypoints3 = np.array(q3.measuredValues)
# ypoints4 = np.array(q4.measuredValues)
# ypoints5 = np.array(q5.measuredValues)


# end = int(np.min([np.max([L1,L2,L3,L4,L5])+5,100]))
# plt.figure()
# a = a_list = list(range(0, end))
# plt.subplot(231)
# plt.hist(ypoints1, bins = a, density = True)
# plt.title('Queue 1')
# plt.subplot(232)
# plt.hist(ypoints2, bins = a, density = True)
# plt.title('Queue 2')
# plt.subplot(233)
# plt.hist(ypoints3, bins = a, density = True)
# plt.title('Queue 3')
# plt.subplot(234)
# plt.hist(ypoints4, bins = a, density = True)
# plt.title('Queue 4')
# plt.subplot(235)
# plt.hist(ypoints5, bins = a, density = True)
# plt.title('Queue 5')
# plt.show()

