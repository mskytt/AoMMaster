#!/usr/bin/env python
"""
    - OIS data processing 
    - Building forward interest rate curves from either cubic splines or utilising matlab-smoothened curves.
    - Computing eigenvectors and principal components

    v0.1 - Mans Skytt (m@skytt.eu)
"""
from __future__ import division
from xlExtract import xlExtract
from h5pyStorage import storeToHDF5, loadFromHDF5
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from datetime import timedelta
import numpy as np
import pandas as pd
import gc

def OIStoZeroCoupon(maturityDates, OISrates):
    ZCrates = np.zeros((1,len(maturityDates))) # Construct array for resulting bootstrapped zero coupon rates
    dateIter = 0
    prevT = 0
    for T in maturityDates:
        sumTerms = 0
        if T <= 1:
            ZCrates[0,dateIter] = 1/T*np.log(1+OISrates[dateIter]*T) # Zero coupon rate
            dateIter += 1
            prevT = T
        else: # always have T > 1 here
            # Find previous payment dates
            prevT = T-1 # Equals previous time for payment.
            while prevT > 0: # Loop through until all payments have been summed and discounted
                index = maturityDates.index(prevT) 
                deltaT = 1*(prevT >= 1) + (T % 1)*(prevT < 1) # Get correct delta T, always = 1 if prevT >= 1
                sumTerms += deltaT*np.exp(-ZCrates[0,index]*prevT) # add term to sum
                prevT -= 1
            ZCrates[0,dateIter] = 1/T*np.log((1+OISrates[dateIter]*1)/(1-OISrates[dateIter]*sumTerms))
            dateIter += 1
    return ZCrates[0,:] # Ugly but needed due to allocation of matrix! gonna try to switch allocation

def ZeroCoupontoForward(maturityDates, ZCrates):
    # manipulation of ZC to get forward rates
    forwardRates = np.zeros(ZCrates.shape)
    forwardRates[0] = ZCrates[0] # First forward rate = first ZC rate 
    i = 1
    while i < ZCrates.size:
        forwardRates[i] = (ZCrates[i]*maturityDates[i] - ZCrates[i-1]*maturityDates[i-1])/(maturityDates[i]-maturityDates[i-1])
        i += 1
    return forwardRates

def runCubicInterp(OISdataVec, matDates):
    # Interpolate with cubic spline given maturity dates and data as a 1D numpy array
    ZCRates = OIStoZeroCoupon(matDates,OISdataVec) # Get zero coupon rates
    forwardRates = ZeroCoupontoForward(matDates,ZCRates) # convert ZC to forward rates
    csFor = CubicSpline(matDates,forwardRates) # Creates cubic spline object that takes time argument
    csZC = CubicSpline(matDates, ZCRates)
    times = np.arange(min(matDates), max(matDates), 1/365)
    csForwardRates = csFor(times) # Splined rates
    csZCRates = csZC(times)
    return forwardRates, csForwardRates, ZCRates, csZCRates, times

def OIStoForMatHelp(OISdataVec, matDates):
    # Helpter function to OISMatToForwardMat(), only returning interpolated forward rates
    forwardRates, csForwardRates, ZCRates, csZCRates, times = runCubicInterp(OISdataVec, matDates)
    return csForwardRates

def OIStoZCMatHelp(OISdataVec, matDates):
    # Helpter function to OISMatToZCMat(), only returning interpolated Zero coupon rates
    forwardRates, csForwardRates, ZCRates, csZCRates, times = runCubicInterp(OISdataVec, matDates)
    return csZCRates

def OISMatToForwardMat(OISdataMat, matDates):
    # Compute matrix of forward rates from OIS data matrix and vector of maturity dates
    forwardMat = np.apply_along_axis(OIStoForMatHelp, 1, OISdataMat, matDates)
    _, _, _, _, times = runCubicInterp(OISdataMat[0,:], matDates)
    return forwardMat, times

def OISMatToZCMat(OISdataMat, matDates):
    # Compute matrix of forward rates from OIS data matrix and vector of maturity dates
    ZCMat = np.apply_along_axis(OIStoZCMatHelp, 1, OISdataMat, matDates)
    _, _, _, _, times = runCubicInterp(OISdataMat[0,:], matDates)
    return ZCMat, times

def runPlotLoop(endRow, startRow, matDates, OISdataMat):
    """
        # Run through vectors
    """
    plt.axis([min(matDates),max(matDates),-0.4,1.7]) # lock axis
    plt.ion()
    row = startRow
    while row < endRow: 
        forwardRates, csForwardRates, ZCRates, csZCRates, times = runCubicInterp(OISdataMat[row,:len(matDates)], matDates)            
        plt.plot(matDates, ZCRates, 'o', label='data')
        plt.plot(times, csZCRates, label='spline')
        #plt.title(OISdataDF.index[row])
        plt.pause(0.1)
        plt.clf()
        row += 1
    return

def runSurfPlot(forwardMat, times):
    """
    Plot a surface of the forward curves.
    the number of days with valid data, for 
        EONIA: 3037, from 2005-08-11 and forward
        FFE <= 1Y mat: 3158, from 2004-12-30 and forward
        FFE <= 2Y mat: 1328, from 2012-01-13 and forward
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    y = np.arange(0,forwardMat.shape[0],1)
    X, Y = np.meshgrid(times, y)

    surf = ax.plot_surface(X, Y, forwardMat, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    #plt.show()

def genEigs(mat):
    # Compute eigen values from matrix
    # Matrix with one observations on each row and one variable in each column 
    covMat = np.cov(mat.T)
    eigVals, eigVecs = np.linalg.eigh(covMat)
    idx = np.argsort(eigVals)[::-1] # Sort in decreasing order
    eigVecs = eigVecs[:,idx] # Arrange vectors accordingly
    eigVals = eigVals[idx]
    sumEigVals = np.sum(eigVals)
    eigPerc = [val/sumEigVals for val in eigVals]
    return eigVals, eigVecs, eigPerc

def genPCs(eigVals, eigVecs, eigPerc, percExpl):
    """
        # Compute principal components given eigen values, eigen vectors, 
        # percentages explained by each eigen vector and the desierd explanation percentage
    """
    i = 0
    cumPerc = 0
    PC = np.zeros(eigVecs.shape)
    while cumPerc < percExpl:
        print eigPerc[i], eigVals[i]
        PC[:,i] = eigVecs[:,i]*np.sqrt(eigVals[i])
        cumPerc += eigPerc[i]
        i += 1
    return PC[:,:i-1], i

def genTimeDelta(dataInd):
    """
        # Compute the time difference, in days, between dates in dataframe Index. 
        # First value = difference between most recent two days.
    """
    timeDeltas = -1*np.diff(dataInd, axis = 0)
    helper = np.vectorize(lambda x: x.days) # Helper function to exctract timedelta days as int 
    timeDeltas = helper(timeDeltas)
    return timeDeltas

def genOISData(readExcel, sheetName, storageFile, matDates, dataCutoff):
    """
        # Extracting OIS data from excel file (special format requried)
        # readExcel - Bool determening if data is to be read from excel files or from HDF5 file
        # storageFile - HDF5 File where to store/load data to/from 
        # matDates - instrument matuirty dates
        # dataCutoff - when to truncate data
    """
    if readExcel:
        OISdata = xlExtract('Data/OIS_data.xlsx',sheetName,0)
        OISdataDF = OISdata.dflinterp # type: pandas.core.frame.DataFrame
        OISdataInd = OISdata.index # type: pandas.tseries.index.DatetimeIndex or nump.ndarray
        OISdataCol = OISdata.columns # type: pandas.indexes.base.Index
        OISdataMat = OISdataDF.values/100 # type: numpy.ndarray, interpolated between points
        OISdataMat = OISdataMat[:,0:len(matDates)]
        OISTimeDelta = genTimeDelta(OISdataInd)
        print 'Extracted data using xlExtract.'
        storeToHDF5(storageFile, 'OISdataMat', OISdataMat)
        storeToHDF5(storageFile, 'OISTimeDelta', OISTimeDelta)
        print 'Stored OIS data matrix to file.'
    else:
        OISdataMat = loadFromHDF5(storageFile,'OISdataMat')
        OISTimeDelta = loadFromHDF5(storageFile, 'OISTimeDelta')
        print 'Read matrix from file.'
    return

def genForwardData(genForward, sheetName, storageFile, matDates, dataCutoff):
    """
        # Compute forward rates from the OIS quotes
        # genForward - Bool determening if data is to be computed from 'scratch' or read from HDF5 file
        # storageFile - HDF5 File where to store/load data to/from 
        # matDates - instrument matuirty dates
        # dataCutoff - when to truncate data
    """
    OISdataMat = loadFromHDF5(storageFile,'OISdataMat')

    if genForward:
        forwardMat, times = OISMatToForwardMat(OISdataMat, matDates)
        forwardMat = forwardMat[:dataCutoff,:]
        forMatDiff = -1*np.diff(forwardMat, axis = 0)
        print 'Generated forward matrices.'
        storeToHDF5(storageFile, 'forwardMat', forwardMat)
        storeToHDF5(storageFile, 'forMatDiff', forMatDiff)
        #storeToHDF5(storageFile, 'times', times)
        print 'Stored forward matrices to file.'
    else:
        forwardMat = loadFromHDF5(storageFile,'forwardMat')
        forMatDiff = loadFromHDF5(storageFile,'forMatDiff')
        #times = loadFromHDF5(storageFile,'times')
        print 'Read forward matrices from file.'
    return

def genZCData(genZC, sheetName, storageFile, matDates, dataCutoff):

        # Compute Zero coupon rates from the OIS quotes
        # genForward - Bool determening if data is to be computed from 'scratch' or read from HDF5 file
        # storageFile - HDF5 File where to store/load data to/from 
        # matDates - instrument matuirty dates
        # dataCutoff - when to truncate data


    OISdataMat = loadFromHDF5(storageFile,'OISdataMat')

    if genZC:
        ZCMat, times = OISMatToZCMat(OISdataMat, matDates)
        ZCMat = ZCMat[:dataCutoff,:]
        ZCMatDiff = -1*np.diff(ZCMat[:dataCutoff,:], axis = 0)
        print 'Generated zero coupon matrices.'
        storeToHDF5(storageFile, 'ZCMat', ZCMat)
        storeToHDF5(storageFile, 'ZCMatDiff', ZCMatDiff)
        storeToHDF5(storageFile, 'times', times)
        print 'Stored zero coupon matrices to file.'
    else:
        ZCMat = loadFromHDF5(storageFile,'ZCMat')
        ZCMatDiff = loadFromHDF5(storageFile,'ZCMatDiff')
        times = loadFromHDF5(storageFile,'times')
        print 'Read zero coupon matrices from file.'
    return

def interpUSG(dataMat, matDates, outputTimes):
    # Interpolate with cubic spline given maturity dates and data as a 1D numpy array

    ZCRates = dataMat # Get zero coupon rates
    cs = CubicSpline(matDates, ZCRates)
    times = np.arange(min(matDates), max(matDates), 1/365)
    csZCRates = cs(times) # Splined rates
    if outputTimes:
        return csZCRates, times
    else:
        return csZCRates

def genUSGGData(genZC, sheetName, storageFile, matDates, dataCutoff):
    
    """
        # Compute forward data from the USGG quotes
        # genZC - Bool determening if data is to be computed from 'scratch' or read from HDF5 file
        # storageFile - HDF5 File where to store/load data to/from 
        # matDates - instrument matuirty dates
        # dataCutoff - when to truncate data
    """


    OISdataMat = loadFromHDF5(storageFile,'OISdataMat')

    if genZC:
        ZCMat = np.apply_along_axis(interpUSG, 1, OISdataMat, matDates, False)
        _, times = interpUSG(OISdataMat[0,:], matDates, True)
        
        ZCMat = ZCMat[:dataCutoff,:]
        ZCMatDiff = -1*np.diff(ZCMat[:dataCutoff,:], axis = 0)
        print 'Generated zero coupon matrices.'
        storeToHDF5(storageFile, 'ZCMat', ZCMat)
        storeToHDF5(storageFile, 'ZCMatDiff', ZCMatDiff)
        storeToHDF5(storageFile, 'times', times)
        print 'Stored zero coupon matrices to file.'
    else:
        ZCMat = loadFromHDF5(storageFile,'ZCMat')
        ZCMatDiff = loadFromHDF5(storageFile,'ZCMatDiff')
        times = loadFromHDF5(storageFile,'times')
        print 'Read zero coupon matrices from file.'
    return

def runGenerateData(readExcel, genForward, genZC, sheetName, storageFile):
    """
        readExcel : bool deciding whether or not data will be read from excel or hdf5
        genForward : bool deciding whether or not forward data will be generated or read from hdf5
        genZC : bool deciding whether or not Zero coupon data will be generated or read from hdf5
        sheetName : Name of sheet
        storageFile: name of file to store/load data to/from
    """
    print 'Started.'
    """
        Define some data based parameters
    """
    EONIAmatDates = [1/52, 2/52,3/52,1/12,2/12,3/12,4/12,5/12,6/12,7/12,8/12,9/12,10/12,11/12,1,15/12,18/12,21/12,2,3,4,5,6,7,8,9,10] #,12,15,20,30,40,50]
    EONIAdataCutoff = 3037 # Number of days with valid data, for EONIA: 3037, from 2005-08-11 and forward
    USGGmatDates = [1/12, 3/12, 6/12, 1, 2, 5, 7, 10]
    USGGdataCutoff = 4100
    FFEmatDates = [1/52, 2/52, 3/52, 1/12, 2/12, 3/12, 4/12, 5/12, 6/12, 7/12, 8/12, 9/12, 10/12, 11/12, 1, 2]
    FFE2YdataCutoff = 1446
    FFE1YdataCutoff = 2800

    if sheetName[0:3] == 'EON':
        matDates = EONIAmatDates
        dataCutoff = EONIAdataCutoff
        print 'EONIA dates defined.'
    elif sheetName[0:3] == 'FFE':
        matDates = FFEmatDates
        dataCutoff = FFE2YdataCutoff
        print 'FFE dates defined.'
    elif sheetName[0:3] == 'USG':
        matDates = USGGmatDates
        dataCutoff = USGGdataCutoff
        print 'USGG dates defined.'

    """
       Read from excel or from .hdf5 file
    """
    genOISData(readExcel, sheetName, storageFile, matDates, dataCutoff)
    """
        Generate forward/Zero-coupon matrix
    """
    if sheetName[0:3] != 'USG':
        genForwardData(genForward, sheetName, storageFile, matDates, dataCutoff)
        genZCData(genZC, sheetName, storageFile, matDates, dataCutoff)
    elif sheetName[0:3] == 'USG':
        genUSGGData(genZC, sheetName, storageFile, matDates, dataCutoff)
    return

def genZCBondPrices(ZCMat,times):
    ZCBondPriceMat = np.exp(-ZCMat*times)
    return ZCBondPriceMat

def genZCBondTS(ZCBondPriceMat,timeDeltas, topDate, startRow, startCol):
    """
    #   Extract the time series for one artificiall zero-coupon bond by utilising:
            - days to maturity = startCol
            - as seen from date corresponding to *startRow* 
        return: 
            - numpy.array with prices from start row until death
            - numpy.array with corresponding dates
    # NOTE: elements in returned arrays are in chronical order with earlies at 0 and most recent at -1
    #       i.e opposite to extracted data
    """
    currCol = startCol
    currRow = startRow

    # Date management
    startDate = topDate - timedelta(days=np.sum(timeDeltas[:startRow])) # get start date
    dateVec = np.array([startDate]) # add start date to vector with all dates

    # Value initialization
    ZCBondPriceVec = ZCBondPriceMat[currRow:currRow+1,currCol] # Initialize by setting first value as np.array
    cumTimeDeltas = timeDeltas[currRow] # add timedeltas to later extract end date
    currCol -= timeDeltas[currRow] 
    currRow -= 1

    # Go forward until bond matures or at last available date
    while currCol >= 0 and currRow >= 0:
        ZCBondPriceVec = np.append(ZCBondPriceVec, ZCBondPriceMat[currRow:currRow+1,currCol]) # Add correct bondprice
        
        dateVec = np.append(dateVec, dateVec[-1]+timedelta(days=timeDeltas[currRow]))
        currCol -= timeDeltas[currRow] # time to maturity is shortened by timeDelta
        currRow -= 1 # Row is decreased by one = move to next timestep

    ZCBondLogReturns = np.diff(np.log(ZCBondPriceVec)) # Computing log returns

    return ZCBondPriceVec, ZCBondLogReturns, dateVec

def genPandaSeries(dataVec, dateVec):
    """
    #   Taskes dataVec and makes it a panda.Series with dateVec
    """
    if dataVec.shape == dateVec.shape:
        _series = pd.DataFrame(data=dataVec, index=dateVec)
        return _series
    else:
        print 'Matrices are of unequal shape \n dataVec: ', dataVec.shape, '\n dateVec: ', dateVec.shape
        return False

def runGenMatlabEigs(genMatlabEigs, storageFile):
    
    MATLABForMatDiff = loadFromHDF5(storageFile, 'MATLABForMatDiff')

    if genMatlabEigs:
        MATLABForEigVals, MATLABForEigVecs, MATLABForEigPerc = genEigs(MATLABForMatDiff)
        print 'Generated Matlab eigen values.'
        storeToHDF5(storageFile, 'MATLABForEigVals', MATLABForEigVals)
        storeToHDF5(storageFile, 'MATLABForEigVecs', MATLABForEigVecs)
        storeToHDF5(storageFile, 'MATLABForEigPerc', MATLABForEigPerc)
        print 'Generated Matlab eigen values to file.'
    else:
        MATLABForEigVals = loadFromHDF5(storageFile,'MATLABForEigVals')
        MATLABForEigVecs = loadFromHDF5(storageFile,'MATLABForEigVecs')
        MATLABForEigPerc = loadFromHDF5(storageFile,'MATLABForEigPerc')
        print 'Read Matlab eigen values from file.'

    MATLABForPCs, MATLABForNumbFactors = genPCs(MATLABForEigVals, MATLABForEigVecs, MATLABForEigPerc, 0.99999)
    print 'Number of factors (MATLAB Forward):', MATLABForNumbFactors
    print 'Generated Matlab forward PCs.'
    storeToHDF5(storageFile, 'MATLABForPCs', MATLABForPCs)
    print 'Stored Matlab forward PCs.'

    return

def runGenMatlab(genMatlab, genMatlabEigs, MATLABForwardMat, sheetName, storageFile):
    """
        # generate corresponting data from matlab generated forward yield curves. Use with output from forward.m
    """

    EONIAdataCutoff = 3000 # Number of days with valid data, for EONIA: 3037, from 2005-08-11 and forward
    FFE2YdataCutoff = 1399
    FFE1YdataCutoff = 2800
    print 'MAtlab shape \n \n', MATLABForwardMat.shape, '\n'

    if sheetName[0:3] == 'EON':
        dataCutoff = EONIAdataCutoff
        print 'EONIA dates defined.'
    elif sheetName[0:3] == 'FFE':
        dataCutoff = FFE2YdataCutoff
        print 'FFE dates defined.'

    if genMatlab:
        MATLABForwardMat = np.flipud(MATLABForwardMat.T)
        MATLABForwardMat = MATLABForwardMat[:dataCutoff,:]
        MATLABForMatDiff = -1*np.diff(MATLABForwardMat, axis = 0)
        # times = loadFromHDF5(storageFile, 'times')
        # runSurfPlot(MATLABForwardMat[:EONIAdataCutoff-1000,:times.shape[0]], times)
        # runSurfPlot(MATLABForMatDiff[:EONIAdataCutoff-1000,:times.shape[0]], times)
        # plt.show()

        print 'Generated Matlab forward matrices.', MATLABForwardMat.shape
        storeToHDF5(storageFile, 'MATLABForMatDiff', MATLABForMatDiff)
        storeToHDF5(storageFile, 'MATLABForwardMat', MATLABForwardMat)
        print 'Stored Matlab forward matrices to file'
    else:
        MATLABForwardMat = loadFromHDF5(storageFile,'MATLABForwardMat')
        MATLABForMatDiff = loadFromHDF5(storageFile, 'MATLABForMatDiff')
        print 'Read Matlab forward matrices from file.'
    
    runGenMatlabEigs(genMatlabEigs, storageFile)

    return

def runGenZCPCs(genZCEigs, ZCMatDiff, storageFile):
    print 'Started runGenZCPCs.'
    """
    #    Generate forward/Zero-coupon eigen values
    """
    if genZCEigs:
        ZCEigVals, ZCEigVecs, ZCEigPerc = genEigs(ZCMatDiff)
        print 'Generated eigen values/vecs from ZC differences'
        storeToHDF5(storageFile, 'ZCEigVals', ZCEigVals)
        storeToHDF5(storageFile, 'ZCEigVecs', ZCEigVecs)
        storeToHDF5(storageFile, 'ZCEigPerc', ZCEigPerc)    
        print 'Stored zero coupon eigen values/vecs to file.'
    else:
        ZCEigVals = loadFromHDF5(storageFile,'ZCEigVals')
        ZCEigVecs = loadFromHDF5(storageFile,'ZCEigVecs')
        ZCEigPerc = loadFromHDF5(storageFile,'ZCEigPerc')
        print 'Read zero coupon eigen values/vecs from file.'

    ZCPCs, ZCNumbFactors = genPCs(ZCEigVals, ZCEigVecs, ZCEigPerc, 0.99999)
    print 'Generated zero-coupon PCs.'
    storeToHDF5(storageFile, 'ZCPCs', ZCPCs)
    print 'Stored zero-coupon PCs.'
    return

def runGenForPCs(genForEigs, forMatDiff, storageFile):
    print 'Started runGenForPCs.'
    """
    #    Generate forward/Zero-coupon principal components from matrix with forward curve differences between days 
    """
    if genForEigs:
        forEigVals, forEigVecs, forEigPerc = genEigs(forMatDiff)
        print 'Generated eigen values/vecs from forward differences.'
        storeToHDF5(storageFile, 'forEigVals', forEigVals)
        storeToHDF5(storageFile, 'forEigVecs', forEigVecs)
        storeToHDF5(storageFile, 'forEigPerc', forEigPerc)
        print 'Stored forward eigen values/vecs to file.'
    else:
        forEigVals = loadFromHDF5(storageFile,'forEigVals')
        forEigVecs = loadFromHDF5(storageFile,'forEigVecs')
        forEigPerc = loadFromHDF5(storageFile,'forEigPerc')
        print 'Read forward eigen values/vecs from file.'

    forPCs, forNumbFactors = genPCs(forEigVals, forEigVecs, forEigPerc, 0.99999)
    print 'Generated forward PCs.'
    storeToHDF5(storageFile, 'forPCs', forPCs)
    print 'Stored forward PCs.'
    return

def run(storageFile, sheetName):
    """
        # Used for testing, debugging and plotting
    """

    # Define dates and numbers
    EONIAmatDates = [1/52, 2/52,3/52,1/12,2/12,3/12,4/12,5/12,6/12,7/12,8/12,9/12,10/12,11/12,1,15/12,18/12,21/12,2,3,4,5,6,7,8,9,10] #,12,15,20,30,40,50]
    FFEmatDates = [1/52, 2/52, 3/52, 1/12, 2/12, 3/12, 4/12, 5/12, 6/12, 7/12, 8/12, 9/12, 10/12, 11/12, 1, 2]
    USGGmatDates = [1/12, 3/12, 6/12, 1, 2, 5, 7, 10]
    #
    #  ADD MANUALLY if used
    #
    OISTopDate = pd.to_datetime('2017-04-20') # Most recent available date, manually added
    
    USGGdataCutoff = 4100
    EONIAdataCutoff = 3000
    FFE2YdataCutoff = 1399
    FFE1YdataCutoff = 2800#3168

    times = loadFromHDF5(storageFile,'times')
    forPCs = loadFromHDF5(storageFile,'forPCs')
    ZCPCs = loadFromHDF5(storageFile,'ZCPCs')
    ZCMat = loadFromHDF5(storageFile,'ZCMat')
    OISTimeDelta = loadFromHDF5(storageFile,'OISTimeDelta')

    # startRow = 800
    # startCol = 365
    # ZCBondPriceMat = genZCBondPrices(ZCMat, times)
    # bondPV, bondLogReturns, dateVec = genZCBondTS(ZCBondPriceMat, OISTimeDelta, OISTopDate, startRow, startCol)
    # runSurfPlot(ZCBondPriceMat[0:2000,:], times)
    # runSurfPlot(ZCMat[0:3000,:], times)
    # plt.plot(dateVec, bondPV)
    # plt.show()
    # plt.plot(dateVec[1:], bondLogReturns)
    # plt.show()

    # forwardMat = loadFromHDF5(storageFile,'forwardMat')
    # forEigVecs = loadFromHDF5(storageFile, 'forEigVecs')
    # forEigPerc = loadFromHDF5(storageFile,'forEigPerc')
    MATLABForwardMat = loadFromHDF5(storageFile,'MATLABForwardMat')
    OISdataMat = loadFromHDF5(storageFile,'OISdataMat')
    MATLABForEigVals = loadFromHDF5(storageFile,'MATLABForEigVals')
    MATLABForEigVecs = loadFromHDF5(storageFile,'MATLABForEigVecs')
    MATLABForEigPerc = loadFromHDF5(storageFile,'MATLABForEigPerc') 
    MATLABForPCs = loadFromHDF5(storageFile,'MATLABForPCs') 
    MATLABForMatDiff = loadFromHDF5(storageFile,'MATLABForMatDiff') 

    # plt.plot(MATLABForEigVecs[:,0:3])
    # # plt.show()
    # plt.plot(MATLABForPCs)
    # plt.show()

    # print MATLABForEigPerc
    runSurfPlot(MATLABForwardMat[:EONIAdataCutoff-1000,:times.shape[0]], times)
    plt.show()
    # runSurfPlot(ZCBondPriceMat[:EONIAdataCutoff,:times.shape[0]], times)
    # runSurfPlot(OISdataMat[:FFE2YdataCutoff,:], FFEmatDates)
    # startRow = 0
    # endRow = 1000
    # runPlotLoop(endRow, startRow, EONIAmatDates, MATLABForwardMat)
    return
