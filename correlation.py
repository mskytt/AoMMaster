#this is a draft for a correlation analysis model

#import packages
#from __future__ import division
from xlExtract import xlExtract
import numpy as np
import pandas as pd
import pdb
import matplotlib.pyplot as plt

# --------------- start program ---------------

#Data extracts

pathsToData = ['Data/OIS_data.xlsx', 'Data/OilFutures.xlsx', 'Data/GoldFutures.xlsx' ] 
sheets = ['EONIA_ASK', 'ReutersICEBCTS', 'ReutersCOMEXGoldTS1', 'ReutersCOMEXGoldTS2', 'ReutersCOMEXGoldTS3', 'ReutersCOMEXGoldTS4']
indexColumns = [0, 0, 0]




class simpleTimeSeries(object):

    def __init__(self, pandaColumn, col_name):

        self.prices = xlExtract.extractData(pandaColumn, col_name,'2017-04-21' , entireTS = True, useLinterpDF = True).dropna()
        try:
            int(self.prices.values[0])
        except ValueError:
            self.prices = self.prices[1:] #remove non-number first row


        self.logReturns =  np.diff(np.log(self.prices.astype(float)))
        self.mean = np.mean(self.logReturns, dtype=np.float64)
        self.calculateGARCH11Variables()
        self. controlStatistics()


        

    def calculateGARCH11Variables(self):

        from arch.univariate import ARCH
        from arch import arch_model

        am = arch_model(self.logReturns) 
        res = am.fit(update_freq=5)
        self.GARCHVolatility = res._volatility
        #fig = res.plot(annualize='D')
        #fig.show()
        print(res.summary())

    def controlStatistics(self):
        print "Standard deviation of log returns"
        print np.std(self.logReturns)



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







pandaColumn =xlExtract(pathsToData[2],sheets[2],indexColumns[2]) #extract one column

col_name = pandaColumn.columns[0] #change to for-loop later

timeSeries = simpleTimeSeries(pandaColumn,col_name)

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



# for module in import_modules:
#     try:
#         tools.install_and_import(module)
#     except ImportError:
#         tools.archInstallSucessfull(False)
#         print module + " not imported"
#
#


