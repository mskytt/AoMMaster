#this is a draft for a correlation analysis model

#import packages
import tools
import numpy as np
import math
import datetime as dt
import scipy
#import_modules = ['scipy.stat', 'math', 'numpy','arch', 'datetime']


#TODO: should we do this on vectors or matrices?
def main_test(data):

    for timeseries in data:
    log_returns = calculate_log_returns(timeseries)
    if archInstallSucessfull:
        sigmas = []
        sigmas.append(getGARCH11Volatilities(log_returns))
        #DCC





# ---------------math tools ---------------
#TODO: change to vector?
def calculate_log_returns(matrix):

    log_returnsMatrix =  math.log((matrix[0:-2,:]/matrix[1:-1,:]) -1)

    return log_returnsMatrix

def getMean(data):
    return np.mean(data)

def diagionalizeVectorToMatrix(vector):
    
    return matrix






# ---------------GARCH ---------------


def getGARCH11volatilities(data):
    from arch.univariate import ARCH, GARCH
    ar.volatility = ARCH(p=5)
    #TODO: vad finns i res? Testa och se hur vollan tas
    res = ar.fit(update_freq=0, disp='off')
    print(res.summary())
    fig = res.plot(annualize='D')

    #testa för att se hur vollan kan returnas
    return res


# --------------- DCC ---------------
def DCC():
#for loop for dates


    #calculate the unconditional mean of each asset
    meanMatrix = getMean(log_ret)




#asset matrix of means
#diagonalize garch volatilities per asset
#log returns per asset
#skapa eta
#hur reda ut Q_tilde? Hur uppskatta alpha och beta
#skapa Q



for module in import_modules:
    try:
        tools.install_and_import(module)
    except ImportError:
        tools.archInstallSucessfull(False)
        print module + " not imported"




