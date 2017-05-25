#!/usr/bin/env python
"""
    Class to extract data from excel files with column indices at first row and 
    row indices specified using indexcolumn in initializer, default is 0.

    v0.1 - Mans Skytt
    ----------------------------------------------------------
    Added linear interpolation of data and changed .ws to .df to represent data frame
    
    v0.2 - Mans Skytt
    ----------------------------------------------------------

"""

import arch, stat, math, os.path, inspect, time
#import matplotlib.pyplot as plt
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
                    self.df = pd.read_excel(xlpath,sheetname,0,0,0,indexcolumn)
                    self.index = self.df.index.to_pydatetime()
                else:
                    self.df = pd.read_excel(xlpath,sheetname)

            else:
                self.df = pd.read_excel(xlpath)

            self.columns = self.df.columns.astype(str) # Convert columns to strings 
            self.dflinterp = self.df.apply(pd.Series.interpolate) # self.df.interpolate(method = 'index') # Linear interpolated Data frame
            self.df.dropna(how = 'all') #drop rows with all nan-values
        else:
            raise IOError(
            "File not found: " + xlpath)

    def extractData(self, LookupColumn, LookupTime, entireTS = False, useLinterpDF = False):
        # Extract values from LookupColumn, type:pandas.core.series.Series, add .value to get: numpy.ndarray
        if not entireTS:
            if LookupTime is str:
                LookupTime = pd.to_datetime(LookupTime) # Convert time string to pandas.tslib.Timestamp 
            # Decide 
            if not useLinterpDF:
                return self.df.loc[LookupTime, LookupColumn]
            else:
                return self.dflinterp.loc[LookupTime, LookupColumn]
        elif not useLinterpDF:
            return self.df.loc[:, LookupColumn]
        else:
            return self.dflinterp.loc[:, LookupColumn]





## Example 
"""
OISdata = xlExtract('Data/OIS_data.xlsx','EONIA_ASK',0)

OISdataDF = OISdata.df
OISdataInd = OISdata.index
OISdataCol =  .columns
LookupColumn = 'EUREON2W='
LookupTime = '2017-04-21'
LookupTime2 = OISdata.index[0:5]
gitDataExtract = OISdataFilled.extractData(LookupColumn, LookupTime, False, True)

print DataExtract
"""