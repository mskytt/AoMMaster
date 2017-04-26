#!/usr/bin/env python

"""
    Class to extract data from excel files with column indices at first row and 
    row indices specified using indexcolumn in initializer, default is 0.

    v0.1 - Mans Skytt

"""

from __future__ import division
import arch, stat, math, random, os.path, inspect, time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime

class xlExtract(object):

    def __init__(self, xlpath, sheetname = None, indexcolumn = 0):
        #Initialize parent class.
        self.xlpath = xlpath

        if os.path.exists(xlpath):
            if sheetname != None:
                self.sheetname = sheetname
                if indexcolumn != None:
                    self.ws = pd.read_excel(xlpath,sheetname,0,0,0,indexcolumn)
                    self.index = self.ws.index.to_datetime() # Convert indices to datetime
                else:
                    self.ws = pd.read_excel(xlpath,sheetname)
            else:
                self.ws = pd.read_excel(xlpath)
            self.columns = self.ws.columns.astype(str) # Convert columns to strings 
        else:
            raise IOError(
            "File not found: " + xlpath)

    def extractData(self, LookupColumn, LookupTime, entireTS = False):
        # Extract values from LookupColumn, type:pandas.core.series.Series, add .value to get: numpy.ndarray
        if not entireTS:
            if LookupTime is str:
                LookupTime = pd.to_datetime(LookupTime) # Convert time string to pandas.tslib.Timestamp
            return self.ws.loc[LookupTime, LookupColumn]
        else:
            return self.ws.loc[:, LookupColumn]


## Example 
OISdata = xlExtract('Data/OIS_data.xlsx','EONIA_ASK',0)

OISdataDF = OISdata.ws
OISdataInd = OISdata.index
OISdataCol = OISdata.columns
LookupColumn = 'EUREON2W='
LookupTime = '2017-04-21'
LookupTime2 = OISdata.index[0:5]

print type(OISdata.extractData(LookupColumn, LookupTime2, True).values)
#OISdataDF['columns'] = OISdataDF['columns'].astype('str')
# OISdataNump = OISdataDF.values #Extract as numpy array
# OISdataDF.index = OISdataDF.index.to_datetime() # Convert indices to datetime
# OISdataDF.columns = OISdataDF.columns.astype(str) # Convert columns to strings 

"""
    dataframe.loc[row,col] => 
"""

#print type(LookupTime)
#print type(OISdataDF.index[0])

#for indexx in OISdataDF.index:
#    print OISdataDF.loc[indexx,LookupColumn]

"""
matDates = [1/52, 2/52,3/52,1/12,2/12,3/12,4/12,5/12,6/12,7/12,8/12,9/12,10/12,11/12,1,15/12,18/12,21/12,2,3,4,5,6,7,8,9,10,12,15,20]
# print matDates
plt.axis([0,20,-0.4,1.7]) # lock axis
plt.ion()
row = 1
""""""
while True:
    while row < 100: 
        plt.plot(matDates, OISdataDF.values[row,1:len(matDates)+1], 'rx')
        plt.title(OISdataDF.index[row])
        plt.pause(0.03)
        plt.clf()
        row+=1
    row = 1
"""
"""
plt.plot(OISdata.columns, 'rx')
plt.ylabel('Some numbers yo')
plt.show()

"""