#!/usr/bin/env python
"""
    - OIS data processing 
    - Building forward interest rate curves
    

    v0.1 - Mans Skytt
"""
from __future__ import division
from xlExtract import xlExtract
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import pandas as pd

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
    ZCrates = OIStoZeroCoupon(matDates,OISdataVec) # Get zero coupon rates
    forwardRates = ZeroCoupontoForward(matDates,ZCrates) # convert ZC to forward rates
    cs = CubicSpline(matDates,forwardRates) # Creates cubic spline object that takes time argument
    times = np.arange(min(matDates), max(matDates), 1/365)
    csForwardRates = cs(times) # Splined rates
    return forwardRates, csForwardRates, times 

def OIStoForMatHelp(OISdataVec, matDates):
    # Helpter function to OISMatToForwardMat(), only returning interpolated forward rates
    forwardRates, csForwardRates, times = runCubicInterp(OISdataVec, matDates)
    return csForwardRates

def OISMatToForwardMat(OISdataMat, matDates):
    # Compute matrix of forward rates from OIS data matrix and vector of maturity dates
    forwardMat = np.apply_along_axis(OIStoForMatHelp, 1, OISdataMat, matDates)
    _, _, times = runCubicInterp(OISdataMat[0,:], matDates)
    return forwardMat, times

def runInterpTest():
    matDatesTest = [0.083333333, 0.166666667, 0.25, 0.5, 0.75, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    OISTest = [-0.00361, -0.00403, -0.00414, -0.00404, -0.00401, -0.00394, -0.00264, -0.00097, 0.00101, 0.00315, 0.0054, 0.008125, 0.009875, 0.01065, 0.0129]    
    ZCrates = OIStoZeroCoupon(matDatesTest,OISTest)
    forwardRates = ZeroCoupontoForward(matDatesTest,ZCrates)
    cs = CubicSpline(matDatesTest,forwardRates)
    times = np.arange(min(matDatesTest), max(matDatesTest), 1/365)
    print forwardRates
    return    

def runPlotLoop(endRow, startRow, matDates, OISdataMat):
    plt.axis([0,max(matDates),-0.4,1.7]) # lock axis
    plt.ion()
    row = startRow
    while row < endRow: 
        forwardRates, csForwardRates, times = runCubicInterp(OISdataMat[row,0:len(matDates)], matDates)            
        plt.plot(matDates, forwardRates, 'o', label='data')
        plt.plot(times, csForwardRates, label='spline')
        plt.title(OISdataDF.index[row])
        plt.pause(0.001)
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
    plt.show()

def genEigs(mat):
    # Matrix with one observations on each row and one variable in each column 
    covMat = np.cov(mat.T)
    eigVals, eigVecs = np.linalg.eigh(covMat)
    idx = np.argsort(eigVals)[::-1] # Sort in decreasing order
    eigVecs = eigVecs[:,idx] # Arrange vectors accordingly
    eigVals = eigVals[idx]
    sumEigVals = np.sum(eigVals)
    eigPerc = [val/sumEigVals for val in eigVals]
    return eigVals, eigVecs, eigPerc

OISdata = xlExtract('Data/OIS_data.xlsx','EONIA_ASK',0)
OISdataDF = OISdata.dflinterp # type: pandas.core.frame.DataFrame
OISdataInd = OISdata.index # type: pandas.tseries.index.DatetimeIndex
OISdataCol = OISdata.columns # type: pandas.indexes.base.Index
OISdataMat = OISdataDF.values/100 # type: numpy.ndarray

matDates = [1/52, 2/52,3/52,1/12,2/12,3/12,4/12,5/12,6/12,7/12,8/12,9/12,10/12,11/12,1,15/12,18/12,21/12,2,3,4,5,6,7,8,9,10] #,12,15,20,30,40,50]
EONIAdataCutoff = 3037 # Number of days with valid data, for EONIA: 3037, from 2005-08-11 and forward
FFE2YdataCutoff = 1328
FFE1YdataCutoff = 315

forwardMat, times = OISMatToForwardMat(OISdataMat, matDates)
eigVals, eigVecs, eigPerc = genEigs(forwardMat[0:EONIAdataCutoff,:])
#plt.plot(times,eigVecs[:,0],times,eigVecs[:,1],times,eigVecs[:,2])
#plt.show()


#runSurfPlot(forwardMat[0:EONIAdataCutoff,:], times)
"""
startRow = 0
endRow = 3000
runPlotLoop(endRow, startRow, matDates, OISdataMat)
"""


