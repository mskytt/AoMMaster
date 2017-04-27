#this is a draft for a correlation analysis model

#import packages
import tools
import scipy
from xlExtract import xlExtract
import numpy as np
import math
import datetime as dt
import scipy
import pdb

#import_modules = ['scipy.stat', 'math', 'numpy','arch', 'datetime']


#TODO: should we do this on vectors or matrices?
def main_test(returns):

    #har borde vi plocka bort nans
    log_returns = calculate_log_returns(returns)
    print "log returns calculated"
    print log_returns
    sigmas = getGARCH11Volatilities(log_returns)

    #DCC



# --------------- FormatData ---------------

#from panda framework to structs 



def pandaMatrixToPandaColumns(pandaData,col):

    key = PandaData.index
    value = pandaData.columns
    # DataFrame([data, index, columns, dtype, copy])

# ---------------math tools ---------------
#takes the log return row-wise
def calculate_log_returns(matrix):
    return np.log(matrix[0:-2,:]) - np.log(matrix[2:,:])

def getMean(data):
    return np.mean(data, dtype=np.float64)

def diagionalizeVectorToMatrix(vector):

    return matrix






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
pandaData =xlExtract('Data/OIS_data.xlsx','EONIA_ASK',0)


main_test(pandaData)
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


