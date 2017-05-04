#this is a draft for a correlation analysis model

#import packages
#from __future__ import division
from xlExtract import xlExtract
import numpy as np
import pandas as pd
import pdb
import matplotlib.pyplot as plt
from math import sqrt
from forwardCurves import OIStoZeroCoupon




class BondorFuturesTimeSeries(object):

    def __init__(self, prices, col_name):

        self.prices = prices 
        try:
            int(self.prices.values[0])
        except ValueError:
            self.prices = self.prices[1:] #remove non-number first row


        self.logReturns =  -100*np.diff(np.log(self.prices.astype(float))) #log returns in percentage
        print "print log returns"
        print self.logReturns
        self.mean_annualized = np.mean(self.logReturns, dtype=np.float64)*sqrt(252) #constant mean
        self.calculateGARCH11Variables()



        

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
        #print(res.summary())

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


#$def createDiagonalMatrixOfVector(vector):








# --------------- start program ---------------

#Data extracts
pathsToData = ['Data/OIS_data.xlsx', 'Data/OilFutures.xlsx', 'Data/GoldFutures.xlsx' ] 
sheets = ['EONIA_MID', 'ReutersICEBCTS', 'ReutersCOMEXGoldTS1', 'ReutersCOMEXGoldTS2', 'ReutersCOMEXGoldTS3', 'ReutersCOMEXGoldTS4']
indexColumns = [0, 0, 0]

#GOLD and OIS-bonds
dfFuturesData =xlExtract(pathsToData[2],sheets[2],indexColumns[2]) #extract one sheet with index column 0 
dfBondsData = xlExtract(pathsToData[0],sheets[0],indexColumns[0]) 


meansFutures = []
GARCHVolatilitiesFutures = []

#BondFutures_pairs = 

#for interest-futures pairs!
#FUTURES
#for i in xrange(len(dfFuturesData.columns)):
for i in xrange(0):
    dfColumnFutures = dfFuturesData.columns[i]


    dfFuturesPrices = xlExtract.extractData(dfFuturesData, dfColumn,'2017-04-21' , entireTS = True, useLinterpDF = True).dropna()
    #dfBondPrices = xlExtract.extractData(dfBondsData, dfColumn,'2017-04-21' , entireTS = True, useLinterpDF = True).dropna()

    timeSeriesFutures = BondorFuturesTimeSeries(dfFuturesPrices, dfColumn)
    meansFutures.append(simpleTimeseries.mean_annualized)
    GARCHVolatilities.append(simpleTimeseries.GARCHVolatility_annualized)


dfMeansFutures = pd.DataFrame(meansFutures,index = dfFuturesData.index)
dfGARCHVolatilitiesFutures =pd.DataFrame(GARCHVolatilitiesFutures,index = dfFuturesData.index) 


#INTEREST RATES


#kolla vilka datum som matchar
for date in dfFuturesData.index:
    dfFuturesData.col



# --------------- DCC ---------------

#create matrix from each individual timestep of each 

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


