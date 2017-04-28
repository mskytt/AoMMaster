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
OISdataMat = OISdataDF.values # type: numpy.ndarray

def OIStoZeroCoupon(maturityDates, OISrates):
    ZCrates = np.zeros((1,len(maturityDates))) # Construct array for resulting bootstrapped zero coupon rates
    sumTerms = np.zeros((1,len(filter(lambda x: x >= 1, matDates)))) # Construct nparray for terms in sum
    dateIter = 0
    sumIter = 0
    prevT = 0
    for T in maturityDates:
        if T <= 1:
            ZCrates[dateIter] = 1/T*np.log(1+OISrates[dateIter]*T) # Zero coupon rate
            if T == 1:
                sumTerms[sumIter] = np.exp(-ZCrates[dateIter]*T) # Term in denominator sum
                sumIter+=1
            dateIter+=1
            prevT = T
        else: # always have T > 1 here
            # Find previous payment dates
            prevT = T-1
            while prevT > 0: # Loop through until all payments have been summed and discounted
                index = maturityDates.index(prevT)
                deltaT = 1*(prevT > 1) + (T % 1)*(prevT < 1) # Get correct delta T, always = 1 if prevT >1
                unEvenSum+= deltaT*np.exp(-ZCrates[index]*prevT) # add term to sum
                prevT-= 1
                sumTerms[sumIter] = np.exp(-ZCrates[dateIter]*T) # Term in denominator sum
                sumumIter+=1

            # Find date diffs between (should be =1 year for all except first)
            # Find Indexes for corresponding zero rates 
            deltaT = T - prevT
            
                
            extraSumTerm = np.exp(-ZCrates[firstPayIndex]*(T-1))
            ZCrates[iterator] = 1/T*np.log((1+OISrates)/(1-OISrates*sum))

print 15/12 % 1

matDates = [1/52, 2/52,3/52,1/12,2/12,3/12,4/12,5/12,6/12,7/12,8/12,9/12,10/12,11/12,1,15/12,18/12,21/12,2,3,4,5,6,7,8,9,10,12,15,20,30,40,50]
print np.zeros((1,len(filter(lambda x: x > 1, matDates)))).shape
y = OISdataMat[0,:]
cs = CubicSpline(matDates,y)
xs = np.arange(1/52, 50, 1/365)
print 1*(0.5-1 > 0) + (3.5 % 1)*(0.5-1 < 0)

plt.plot(matDates, y, 'o', label='data')
plt.plot(xs, cs(xs), label='spline')
plt.legend(loc='lower left')
plt.show()