#!/usr/bin/env python
"""
    - Reworked correlation

    v0.1 - Mans Skytt
"""
from __future__ import division
from xlExtract import xlExtract
import numpy as np
import pandas as pd
import warnings
import scipy
from h5pyStorage import storeToHDF5, loadFromHDF5
import matplotlib.pyplot as plt
from forwardCurves import runSurfPlot, OIStoZeroCoupon, genZCBondPrices, genPandaSeries, genTimeDelta
import math
def matchIndexes(df1, df2):
    """
    # Match indexes in two dataframes, only keep intersection of rows
    """
    intersectIndex = df1.index.intersection(df2.index)
    df1new = df1.reindex(intersectIndex)
    df2new = df2.reindex(intersectIndex)
    return df1new, df2new

def genBondTSfromDf(BondDf):
    """
    #   Extract the time series for one artificiall zero-coupon bond. 
    #   Return: time series with most recent at index 0
    """
    timeDeltas = genTimeDelta(BondDf.index.to_pydatetime())
    DTM = np.sum(timeDeltas) # Sum all timeDeltas to find maturity
    currDTM = 0
    currRow = 0
    bondPriceMat = BondDf.values

    # Value initialization
    bondTS = bondPriceMat[currRow:currRow+1,currDTM] # Initialize by setting first value as np.array
    currDTM += timeDeltas[currRow] 
    currRow += 1

    # Go forward until bond matures or at last available date
    while currDTM <= DTM and currRow <= timeDeltas.shape[0]:
        bondTS = np.append(bondTS, bondPriceMat[currRow,currDTM]) # Add correct bondprice

        if currRow != timeDeltas.shape[0]: # to ensure index not out of range
            currDTM += timeDeltas[currRow] # time to maturity is shortened by timeDelta
        currRow += 1 # Row is increased by one = move back one timestep
    
    logReturns = -1*np.diff(np.log(bondTS)) # Computing log returns
    return bondTS, logReturns


#Data extraction parameters
def genStats(pathToSaveFile, activeCommodity):
    """
        saves EWMA in 'pathToSaveFile' as 'EWMAcovData' + 'column', e.g. 'EWMAcovDataGCH7^1' 
    """
    commodityNumbs = {'Oil' : 1, 'Gold' : 2, 'Power' : 3} # Dict of commidity numbers
    commodityNumb = commodityNumbs[activeCommodity]  # 1 = oil, 2 = gold, 3 = power
    pathsToData = ['Data/OIS_data.xlsx', 'Data/OilFutures.xlsx', 'Data/GoldFutures.xlsx', 'Data/PowerFutures.xlsx' ] 
    

    OISsheets = ['EONIA_MID', 'FFE_MID']
    oilSheets = ['ReutersICEBCTS'] 
    goldSheets = ['ReutersCOMEXGoldTS1', 'ReutersCOMEXGoldTS2', 'ReutersCOMEXGoldTS3']
    powerSheets = ['ReutersNordpoolPowerTS_1', 'ReutersNordpoolPowerTS_2']
    
    sheets = {1 : oilSheets, 2 : goldSheets, 3 : powerSheets} # Dict of sheet arrays

    # Have to cut data to only use ''
    EONIAdataCutoff = 3000
    FFE2YdataCutoff = 1399

    # Select correct paths and other jibberish
    if commodityNumb == 3: # Only power use EONIA
        storageFile = 'EONIAmid.hdf5' # Name of file where data is to be/ is currently stored
        activOIS = 'EONIA'
        OISsheet = OISsheets[0]
        dataCutoff = EONIAdataCutoff
    else:
        storageFile = 'FFEmid.hdf5' # Name of file where data is to be/ is currently stored
        activOIS = 'FFE'
        OISsheet = OISsheets[1]
        dataCutoff = FFE2YdataCutoff

    print 'Storage file: ', storageFile
    print 'Commodity type:', activeCommodity,' with path: ', pathsToData[commodityNumb]

    """
        # Future and OIS data extraction
        # dfFutureTS: time series of ''active'' future
    """
    # Allocate matrices for statistics
    corrCoefPearsonVec = np.array([])
    corrCoefSpearmanVec = np.array([])
    covEntireTSVec = np.array([])
    pValSpearmanVec = np.array([])
    stackedLogReturns = np.array([])
    instrumentVec = np.array([])
    maturityVec = np.array([])
    numbInstruments = 0

    for sheet_ in sheets[commodityNumb]:
        print 'In sheet: ', sheet_
        futuresDataMat = xlExtract(pathsToData[commodityNumb], sheet_, 0) #extract one sheet with index column 0 
        dfFuturesData = futuresDataMat.df
        ZCData = xlExtract(pathsToData[0],OISsheet, 0) # Load from data frame to get indexes and columns
        dfZCData = ZCData.dflinterp[:dataCutoff]
        # Load data to input into dataframe of bonds
        ZCMat = loadFromHDF5(storageFile,'ZCMat')
        times = loadFromHDF5(storageFile,'times')

        dfZCMat = pd.DataFrame(data=ZCMat[:dataCutoff,:], index=ZCData.index[:dataCutoff], columns=times) # Dataframe of ZC matrix to use for date-matching
        i = 0
        for column in dfFuturesData.columns:
            #print 'At instrument: ', column, ' (', sheet_, ')'
            dfFutureTS = xlExtract.extractData(futuresDataMat, column, '', entireTS = True, useLinterpDF = False).dropna()
            
            # If time series is empty, skip iteration
            if dfFutureTS.empty:
                print column, 'does not contain any data.'
                continue
            # If future is alive longer than the furthest OIS-contract, skip iteration
            if np.sum(genTimeDelta(dfFutureTS.index.to_pydatetime())) > times.shape[0]:
                print column, 'has a too long time series for OIS data.'
                continue

            matchedDfZCMat, matchedDfFutureTS  = matchIndexes(dfZCMat, dfFutureTS) # Matching data at index
    
            # If time series are not overlapping, skip iteration
            if matchedDfZCMat.empty:
                print 'No overlapping dates, ', column, ' (', sheet_, ')'
                continue

            ZCBondMat = genZCBondPrices(matchedDfZCMat.values, times) # Generate bond prices
            ZCBondDfMat = pd.DataFrame(data=ZCBondMat, index=matchedDfZCMat.index, columns=times) # Create zero-coupon bond data frame

            # Compute log-returns and time series
            futureTS = matchedDfFutureTS.values
            ZCBondTS, ZCBondLogReturns = genBondTSfromDf(ZCBondDfMat)
            
            """
            #   IGNORE IF NEGATIVE VALUES IN QUOTES! TO FIX!!
            """
            warnings.filterwarnings("error")
            try:
                futureLogReturns = -1*np.diff(np.log(futureTS)) # Computing log returns
            except RuntimeWarning:
                #print 'Returns are negative, ignoring:', column
                continue
            warnings.filterwarnings("always")
            
            logReturnMatRows = np.column_stack((futureLogReturns, ZCBondLogReturns)).T
            logReturnMatCol = np.column_stack((futureLogReturns, ZCBondLogReturns))
            futureLogReturnsDF = pd.DataFrame(data=futureLogReturns, index=matchedDfFutureTS.index[1:])
            ZCBondLogReturnsDF = pd.DataFrame(data=ZCBondLogReturns, index=matchedDfZCMat.index[1:])
            logReturnMatColDF = pd.DataFrame(data=logReturnMatCol, index=matchedDfZCMat.index[1:])
            if stackedLogReturns.size == 0: # Stack all log-returns
                stackedLogReturns = logReturnMatCol # Initialize first
            else: 
                stackedLogReturns = np.vstack((stackedLogReturns,logReturnMatCol)) # vertical stack new returns below, [:,0] = future logreturns, [:,1] = bond log returns
            maturity = matchedDfFutureTS.index[0] - matchedDfFutureTS.index[-1]

            """ 
            #   Compute EWMA covariance. alpha = 1-lambda, given risk metrics recommendation (lambda=0.94) alpha is set to 0.06 
            """
            ewmCovDF = futureLogReturnsDF.ewm(alpha=0.06,min_periods=0,adjust=True).cov(bias=False,other=ZCBondLogReturnsDF,pairwise=False)
            EWMAcovData = ewmCovDF.values
            EWMAcovDates = ewmCovDF.index
            
            storeToHDF5(pathToSaveFile, 'EWMAcovData'+column, EWMAcovData)

            # Correlation and covariance
            covMatEntireTS = np.cov(logReturnMatRows)*252 # takes covariance with variables on rows, return covariance matrix
            corrCoefPearson = np.corrcoef(logReturnMatRows) # Pearson product moment correlation coefficients
            corrCoefSpearman, pValSpearman = scipy.stats.spearmanr(ZCBondLogReturns, futureLogReturns)

            corrCoefPearsonVec = np.append(corrCoefPearsonVec, corrCoefPearson[0,1]) # Take only one value (2x2 mat)
            covEntireTSVec = np.append(covEntireTSVec, covMatEntireTS[0,1]) # Take only one value (2x2 mat)
            corrCoefSpearmanVec = np.append(corrCoefSpearmanVec, corrCoefSpearman)
            pValSpearmanVec = np.append(pValSpearmanVec, pValSpearman)
            instrumentVec = np.append(instrumentVec, column.encode('ascii','ignore')) 
            maturityVec = np.append(maturityVec, maturity.days)
            numbInstruments += 1 
            if np.abs(np.amax(futureLogReturns)) > 2:
                print column, 'has abnormal log-returns (abs > 200%).'
            # print covMatEntireTS, '\n', corrCoefPearson, '\n', corrCoefSpearman, pValSpearman

    storeToHDF5(pathToSaveFile, activeCommodity+'InstrumentVec', instrumentVec)
    storeToHDF5(pathToSaveFile, activeCommodity+'MaturityVec', maturityVec)
    storeToHDF5(pathToSaveFile, activeCommodity+'CovEntireTSVec', covEntireTSVec)   
    storeToHDF5(pathToSaveFile, activeCommodity+'CorrCoefPearsonVec', corrCoefPearsonVec)
    storeToHDF5(pathToSaveFile, activeCommodity+'CorrCoefSpearmanVec', corrCoefSpearmanVec)
    storeToHDF5(pathToSaveFile, activeCommodity+'pValSpearmanVec', pValSpearmanVec)
    storeToHDF5(pathToSaveFile, activeCommodity+'StackedLogReturns', stackedLogReturns)
    print 'Number of instruments evaluated:', numbInstruments
    return


pathToSaveFile = 'stats.hdf5'
activeCommodity = 'Power'
# genStats(pathToSaveFile, activeCommodity)

# stackedLogReturns = loadFromHDF5(pathToSaveFile, activeCommodity+'StackedLogReturns')
# plt.scatter(stackedLogReturns[:,0], stackedLogReturns[:,1], marker='o', alpha=0.3)
# plt.show()
# indexesZeroElements = np.where(stackedLogReturns[:,0] == 0)[0]
maturityVec = loadFromHDF5(pathToSaveFile, activeCommodity+'MaturityVec')
covEntireTSVec = loadFromHDF5(pathToSaveFile, activeCommodity+'CovEntireTSVec')
corrCoefPearsonVec = loadFromHDF5(pathToSaveFile, activeCommodity+'CorrCoefPearsonVec')
corrCoefSpearmanVec = loadFromHDF5(pathToSaveFile, activeCommodity+'CorrCoefSpearmanVec')
instrumentVec = loadFromHDF5(pathToSaveFile, activeCommodity+'InstrumentVec')


# fig, ax = plt.subplots()
# ax.set_xticks(range(0,instrumentVec.shape[0]))
# ax.set_xticklabels(instrumentVec, rotation='vertical', fontsize=10)
# ax.plot(range(0,instrumentVec.shape[0]),covEntireTSVec)
# plt.show()
# plt.plot(corrCoefPearsonVec)
# plt.show()
# plt.plot(corrCoefSpearmanVec)
# plt.show()

# covEntireTSVec = loadFromHDF5(pathToSaveFile, 'covEntireTSVec')
# corrCoefPearsonVec = loadFromHDF5(pathToSaveFile, 'corrCoefPearsonVec')
# corrCoefSpearmanVec = loadFromHDF5(pathToSaveFile, 'corrCoefSpearmanVec')
# pValSpearmanVec = loadFromHDF5(pathToSaveFile, 'pValSpearmanVec')


# plt.hist(corrCoefPearsonVec, bins=50)
# plt.show()  
# plt.hist(corrCoefSpearmanVec, bins=50)
# plt.show()
# plt.hist(covEntireTSVec, bins=50)
# plt.show()

# plt.scatter(ZCBondLogReturns, futureLogReturns)
# plt.show()