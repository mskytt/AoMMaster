#!/usr/bin/env python
"""
    - OIS data processing 
    - Building forward interest rate curves

    v0.1 - Mans Skytt
"""

from xlExtract import xlExtract
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OISdata = xlExtract('Data/OIS_data.xlsx','EONIA_ASK',0)
OISdataFilled = xlExtract('Data/OIS_data.xlsx','EONIA_ASK',0)

OISdataDF = OISdata.dflinterp # type: pandas.core.frame.DataFrame
OISdataInd = OISdata.index # type: pandas.tseries.index.DatetimeIndex
OISdataCol = OISdata.columns # type: pandas.indexes.base.Index
OISdataMat = OISdataDF.values # type: numpy.ndarray

