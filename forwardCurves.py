#!/usr/bin/env python
"""
    - OIS data processing 
    - Building forward interest rate curves

    v0.1 - Mans Skytt
"""

from xlExtract import xlExtract
import matplotlib.pyplot as plt

OISdata = xlExtract('Data/OIS_data.xlsx','EONIA_ASK',0)

OISdataDF = OISdata.ws # type: pandas.core.frame.DataFrame
OISdataInd = OISdata.index # type: pandas.tseries.index.DatetimeIndex
OISdataCol = OISdata.columns # type: pandas.indexes.base.Index
OISdataMat = OISdataDF.values # type: numpy.ndarray

LookupColumn = 'EUREON2W='
LookupTime = '2017-04-21'
LookupTime2 = OISdata.index[0:5]

InterpolTS = OISdata.extractData(LookupColumn, LookupTime2, True)
NonNAindex <- which(!is.na(InterpolTS))
firstNonNA <- max(NonNAindex)


#np.interp(x, xp, fp, period=360)
print type(OISdataDF)

#plt.plot()
#plt.show() 