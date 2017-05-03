#this is a draft for a correlation analysis model

#import packages
#from __future__ import division
from xlExtract import xlExtract
import numpy as np
import pandas as pd
import pdb
import matplotlib.pyplot as plt
from math import sqrt




class simpleTimeSeries(object):

    def __init__(self, prices, col_name):

        self.prices = prices 
        try:
            int(self.prices.values[0])
        except ValueError:
            self.prices = self.prices[1:] #remove non-number first row


        self.logReturns =  -100*np.diff(np.log(self.prices.astype(float))) #log returns in percentage
        print "print log returns"
        print self.logReturns
        self.mean = np.mean(self.logReturns, dtype=np.float64)
        self.calculateGARCH11Variables()
        self. controlStatistics()


        

    def calculateGARCH11Variables(self):

        from arch.univariate import ARCH
        from arch import arch_model

        am = arch_model(self.logReturns) 
        res = am.fit(update_freq=5)
        self.GARCHVolatility = res._volatility
        self.GARCHVolatility_annualized = self.GARCHVolatility *sqrt(252)
        print self.GARCHVolatility_annualized 
        #fig = res.plot(annualize='D')
        #fig.show()
        print(res.summary())

    def controlStatistics(self):
        print "Standard deviation of log returns"
        print np.std(self.logReturns)*sqrt(252)



# --------------- DCC ---------------
# def DCC():
# #for loop for dates


#     #calculate the unconditional mean of each asset
#     meanMatrix = getMean(log_ret)



def plotDistribution(log_returns):
    plt.hist(log_returns)
    plt.title("Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()









# --------------- start program ---------------

#Data extracts

pathsToData = ['Data/OIS_data.xlsx', 'Data/OilFutures.xlsx', 'Data/GoldFutures.xlsx' ] 
sheets = ['EONIA_ASK', 'ReutersICEBCTS', 'ReutersCOMEXGoldTS1', 'ReutersCOMEXGoldTS2', 'ReutersCOMEXGoldTS3', 'ReutersCOMEXGoldTS4']
indexColumns = [0, 0, 0]

pandaData =xlExtract(pathsToData[2],sheets[2],indexColumns[0]) #extract one sheet with index column 0 


#for i in xrange(len(pandaData.columns)):
for i in xrange(0):
    pandaColumn = pandaData.columns[i]
    prices = xlExtract.extractData(pandaData, pandaColumn,'2017-04-21' , entireTS = True, useLinterpDF = True).dropna()
    timeSeries = simpleTimeSeries(prices, pandaColumn)


#plotDistribution(timeSeries.log_returns)
#print timeSeries.GARCHmu
#print timeSeries.mean
    #DCC

#asset matrix of means
#diagonalize garch volatilities per asset
#log returns per asset
#skapa eta
#hur reda ut Q_tilde? Hur uppskatta alpha och beta
#skapa Q


