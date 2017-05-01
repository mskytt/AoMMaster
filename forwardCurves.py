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
import numpy as np
import pandas as pd

OISdata = xlExtract('Data/OIS_data.xlsx','EONIA_ASK',0)

OISdataDF = OISdata.dflinterp # type: pandas.core.frame.DataFrame
OISdataInd = OISdata.index # type: pandas.tseries.index.DatetimeIndex
OISdataCol = OISdata.columns # type: pandas.indexes.base.Index
OISdataMat = OISdataDF.values/100 # type: numpy.ndarray

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
    forwardRate = 0 # manipulation of ZC to get forward rates
    deltaTs = [j-i for i, j in zip(maturityDates[:-1], maturityDates[1:])]



matDates = [1/52, 2/52,3/52,1/12,2/12,3/12,4/12,5/12,6/12,7/12,8/12,9/12,10/12,11/12,1,15/12,18/12,21/12,2,3,4,5,6,7,8,9,10] #,12,15,20,30,40,50]

y = OIStoZeroCoupon(matDates,OISdataMat[0,0:len(matDates)])
cs = CubicSpline(matDates,y)
xs = np.arange(min(matDates), max(matDates), 1/365)
print ZeroCoupontoForward(matDates,y)

plt.axis([0,max(matDates),-0.4,1.7]) # lock axis
plt.ion()
row = 3000
while row < 3500: 
    y = OIStoZeroCoupon(matDates,OISdataMat[row,0:len(matDates)])
    cs = CubicSpline(matDates,y)
    xs = np.arange(min(matDates), max(matDates), 1/365)
        
    plt.plot(matDates, y, 'o', label='data')
    plt.plot(xs, cs(xs), label='spline')
    plt.title(OISdataDF.index[row])
    plt.pause(0.001)
    plt.clf()
    row += 1
