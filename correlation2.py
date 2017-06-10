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
import matplotlib.mlab as mlab

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
    

    OISsheets = ['EONIA_MID', 'FFE_MID', 'USGG_MID']
    oilSheets = ['ReutersICEBCTS'] 
    goldSheets = ['ReutersCOMEXGoldTS1', 'ReutersCOMEXGoldTS2', 'ReutersCOMEXGoldTS3']
    powerSheets = ['ReutersNordpoolPowerTS_1', 'ReutersNordpoolPowerTS_2']
    
    sheets = {1 : oilSheets, 2 : goldSheets, 3 : powerSheets} # Dict of sheet arrays

    # Have to cut data to only use ''
    EONIAdataCutoff = 3000
    FFE2YdataCutoff = 1399
    USGGdataCutoff = 4100
    # Select correct paths and other jibberish
    if commodityNumb == 3: # Only power use EONIA
        storageFile = 'EONIAmid.hdf5' # Name of file where data is to be/ is currently stored
        activOIS = 'EONIA'
        OISsheet = OISsheets[0]
        dataCutoff = EONIAdataCutoff
    else:
        storageFile = 'USGGmid.hdf5' # Name of file where data is to be/ is currently stored
        activOIS = 'USGG'
        OISsheet = OISsheets[2]
        dataCutoff = USGGdataCutoff

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
        ZCData = xlExtract(pathsToData[0], OISsheet, 0) # Load from data frame to get indexes and columns
        dfZCData = ZCData.dflinterp[:dataCutoff]
        
        # Load data to input into dataframe of bonds
        originalZCMat = loadFromHDF5(storageFile,'ZCMat')
        oiriginalTimes = loadFromHDF5(storageFile,'times')
        # Extend to include all times down to 1 day
        extraTimes = np.arange(1/365,oiriginalTimes[0]-1/365,1/365)
        times = np.append(extraTimes, oiriginalTimes)
        
        extraSteps = extraTimes.shape[0]
        extendedZCMat = np.repeat(originalZCMat[:,0:1],extraSteps, axis=1)
        ZCMat = np.column_stack((extendedZCMat, originalZCMat))

        dfZCMat = pd.DataFrame(data=ZCMat[:dataCutoff,:], index=ZCData.index[:dataCutoff], columns=times) # Dataframe of ZC matrix to use for date-matching

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

def meansInBins(valuesVec, numbOfEach, bins):
    valuesInEachBin = np.

    for i in range(bins):


    return newValuesVec, newNumbofEach


pathToSaveFile = 'stats.hdf5'
activeCommodity = 'Gold'
doPlott = False
genStats(pathToSaveFile, activeCommodity)

maturityVec = loadFromHDF5(pathToSaveFile, activeCommodity+'MaturityVec')
covEntireTSVec = loadFromHDF5(pathToSaveFile, activeCommodity+'CovEntireTSVec')
corrCoefPearsonVec = loadFromHDF5(pathToSaveFile, activeCommodity+'CorrCoefPearsonVec')
corrCoefSpearmanVec = loadFromHDF5(pathToSaveFile, activeCommodity+'CorrCoefSpearmanVec')
instrumentVec = loadFromHDF5(pathToSaveFile, activeCommodity+'InstrumentVec')
stackedLogReturns = loadFromHDF5(pathToSaveFile, activeCommodity+'StackedLogReturns')

uniqueMaturities = np.unique(maturityVec) 
statArray = np.vstack((covEntireTSVec, corrCoefPearsonVec, corrCoefSpearmanVec))
statName = np.array(['Covariance', 'Pearson Correlation', 'Spearman Correlation'])
for stat, name in zip(statArray, statName):
    covMean = np.array([])
    covMedian = np.array([])
    spearmanMean = np.array([])
    spearmanMedian = np.array([])
    pearsonMean = np.array([])
    pearsonMedian = np.array([])
    numbWithMaturity = np.array([])
    for maturity in uniqueMaturities:
        tempCovariances = stat[np.where(maturityVec == maturity)]
        tempNumb = tempCovariances.size 
        covMean = np.append(covMean, tempCovariances.mean()) # Extract covariances for the corresponding maturity and store mean
        covMedian = np.append(covMedian, np.median(tempCovariances)) # Extract covariances for the corresponding maturity and store median
        numbWithMaturity = np.append(numbWithMaturity, tempNumb) # Keep track of amout of instruments with the maturity

    """
    #   Plot mania
    """
    if doPlott:
        fig, ax1 = plt.subplots()
        ax1.bar(left=uniqueMaturities+np.append(np.diff(uniqueMaturities),1)/2, height=numbWithMaturity, width=np.append(np.diff(uniqueMaturities), 1), color='g', label='Number of instruments with maturity')
        #ax1.plot(uniqueMaturities, numbWithMaturity, 'g-', label='Number of instruments with maturity')
        ax1.set_ylabel('Amount', color='g')
        ax1.tick_params('y', colors='g')

        ax2 = ax1.twinx()
        ax2.plot(uniqueMaturities, covMean, 'b^', label='Mean '+ name +' for maturity')
        ax2.plot(uniqueMaturities, covMedian, 'rv', label='Median '+ name +' for maturity')
        ax2.set_xlabel('Maturity (days)')
        ax2.set_ylabel(name)

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1+h2, l1+l2, loc=1)
        fig.tight_layout()
        plt.title(name+' ('+activeCommodity+')')
if doPlott:
    plt.show()

# plt.plot(uniqueMaturities, covMean, uniqueMaturities, covMedian)
# plt.show()

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

plt.figure(1)
plt.hist(stackedLogReturns[:,0], bins=100, normed=True)
plt.xlim((min(stackedLogReturns[:,0]), max(stackedLogReturns[:,0])))

mean = np.mean(stackedLogReturns[:,0])
variance = np.var(stackedLogReturns[:,0])
sigma = np.sqrt(variance)
x = np.linspace(min(stackedLogReturns[:,0]), max(stackedLogReturns[:,0]), 100)
plt.plot(x, mlab.normpdf(x, mean, sigma))

plt.show()

# plt.hist(corrCoefSpearmanVec, bins=50)
# plt.show()
# plt.hist(covEntireTSVec, bins=50)
# plt.show()

# plt.scatter(ZCBondLogReturns, futureLogReturns)
# plt.show()
