#this is a draft for a correlation analysis model

#import packages
#from __future__ import division
from xlExtract import xlExtract
import numpy as np
import pandas as pd
import pdb






#TODO: should we do this on vectors or matrices?
def main_test(pandaData,col_name):

    prices = xlExtract.extractData(pandaData, col_name,'2017-04-21' , entireTS = True, useLinterpDF = True)
    prices = prices.dropna() #remove nan-rows

    try:
     int(prices.values[0])
    except ValueError:
        prices = prices[1:] #remove non-number first row



    print prices.values
    log_returns = calculate_log_returns(prices.values)
    print "log returns calculated"
    #print log_returns
    pdb.set_trace()
    sigmas = getGARCH11Volatilities(log_returns)
    print sigmas
    #DCC



# --------------- FormatData ---------------

#from panda framework to structs 


def pandaMatrixToPandaColumns(pandaData):

    key = pandaData.index
    value = pandaData.columns[col]
    returns = pd.DataFrame([value, key])
    return returns





# ---------------math tools ---------------
def calculate_log_returns(prices):
    return np.log(float(prices[0:-2])) - np.log(float(prices[1:-1]))

# def getMean(data):
#     return np.mean(data, dtype=np.float64)

# def diagionalizeVectorToMatrix(vector):

#     return matrix






# ---------------GARCH ---------------



def getGARCH11Volatilities(log_returns):

    from arch.univariate import ARCH
    from arch import arch_model

    am = arch_model(log_returns) 
    res = am.fit(update_freq=5)

    print(res.summary())

    # ar.volatility = ARCH(p=5)
    # #TODO: vad finns i res? Testa och se hur vollan tas
    # res = ar.fit(update_freq=0, disp='off')
    # print(res.summary())
    # fig = res.plot(annualize='D')

    # #test to see how the vol could be returned
    # return res


# --------------- DCC ---------------
def DCC():
#for loop for dates


    #calculate the unconditional mean of each asset
    meanMatrix = getMean(log_ret)




# --------------- start program ---------------

#Data extracts

pathsToData = ['Data/OIS_data.xlsx', 'Data/OilFutures.xlsx', 'Data/GoldFutures.xlsx' ] 
sheets = ['EONIA_ASK', 'ReutersICEBCTS', 'ReutersCOMEXGoldTS1', 'ReutersCOMEXGoldTS2', 'ReutersCOMEXGoldTS3', 'ReutersCOMEXGoldTS4']
indexColumns = [0, 0, 0]

#extract one column
pandaColumn =xlExtract(pathsToData[2],sheets[2],indexColumns[2])
#change to for-loop later
 # for col_index in xrange(0,len(pandaData.columns)):
 #   
col_name = pandaColumn.columns[0]


main_test(pandaColumn, col_name)


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


