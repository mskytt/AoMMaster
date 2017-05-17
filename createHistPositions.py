from xlExtract import xlExtract
import pandas as pd
import numpy as np
from h5pyStorage import storeToHDF5, loadFromHDF5
import os.path
from pdb import set_trace
import datetime






def getInterestRateDates(EONIA = True, FFE = False):
	if EONIA:
		pathToData = 'Data/OIS_Data.xlsx'
		sheet = 'EONIA_MID'
		maturitiesInterest = loadFromHDF5('EONIAask.hdf5','times') #maturities, column wise of interest data
		datesInterest = pd.read_excel(pathToData,sheet ,parse_cols = [0]) #dates, row wise of interest rate data
		datesInterest['dates'] = pd.to_datetime(datesInterest['dates'])  # pandas.core.frame.DataFrame of 'numpy.datetime64'> elements
		return datesInterest
	else: #TODO
		return []

def getInterestRates():
	return loadFromHDF5('EONIAask.hdf5','MATLABForwardMat') #interest rates data, type



 #----------USAGE --------

def useData(dfFuturesData,ForwarRatesdMat,datesInterest):
	interestRates = getCorrespondingInterestRates(dfFuturesData,ForwarRatesdMat,datesInterest)




 
def getCorrespondingInterestRates(dfFuturesData,ForwarRatesdMat,datesInterest):
	column = dfFuturesData.columns[0] #for column in dfFuturesData.columns:
	datesFutures = dfFuturesData[column].dropna().index #<class 'pandas.tseries.index.DatetimeIndex'
	datesFutures = np.array(datesFutures, dtype = 'datetime64[ns]') #<type 'numpy.ndarray'> of <numpy.datetime64'> elements
	interestRates = []
	ForwarRatesdMat_index = []
	_datesFutures = []
	_unusedDatesFutures = []
	timeToMats = []
	timeToMat = len(datesFutures)
	print "finding corresponding interest rates for future " + column
	i = 0
	for date in datesFutures:
		i += 1
		if datesInterest[datesInterest['dates'].values == date].index.tolist(): #spara 'ven vilka datum som finns'
			#print datesInterest[datesInterest['dates'].values == date].index.tolist()
			ForwarRatesdMat_index.append(datesInterest[datesInterest['dates'].values == date].index.tolist()[0])

			_datesFutures.append(date) #only save those who have a corresponding interest rate date
			timeToMats.append(timeToMat - i) 
		else: 
			 _unusedDatesFutures.append(datesInterest[datesInterest['dates'].values == date].index.tolist())

	return ForwarRatesdMat[ForwarRatesdMat_index,timeToMat]



def getForwardPosition(strikePrice, maturityPrice):
	return  maturityPrice - strikePrice

def getFuturesPosition(futurePrices,interestRates):
	strike = futuresPrices[0]

	for i in xrange(len(futuresPrices),interestRates):
		timeToMat = getMaturity(col_name,dates)
		futuresPosition += (futuresPrices.values[i] - strike)*interestRates.values[i]*timeToMat
	return futuresPosition

def getMaturity_inDays(name,dates):
	return  (dates.loc[name, u'FINAL SETTLE DATE'] -  dates.loc[name, u'FUT.1ST PRICE DATE']).days

	
# --------------- start program ---------------

# CHOOSE COMMODITY
gold = True
aluminimum = False
oil = False
power = False


# -----------------------------------------------
if gold:
	#interest rate dates and data
	datesInterest = getInterestRateDates(EONIA = True, FFE = False)
	ForwarRatesdMat = getInterestRates()

	#futures dates and data
	pathToData =  'Data/GoldFutures.xlsx'
	sheets = ['ReutersCOMEXGoldTS1', 'ReutersCOMEXGoldTS2', 'ReutersCOMEXGoldTS3']
	indexColumn = 0
	dfFuturesData = pd.DataFrame()
	for sheet in sheets[0:1]:
		xlsFuturesData = xlExtract(pathToData,sheet,indexColumn)
		dfFuturesData = xlExtract.extractData(xlsFuturesData, xlsFuturesData.columns,'2017-04-21',  entireTS = True, useLinterpDF = False).dropna(how = 'all')
		print "sheet " + str(sheet) + " extracted"	
		useData(dfFuturesData,ForwarRatesdMat,datesInterest)
		

if aluminimum:
	pass

if oil:
	pathToData = 'Data/OilFutures.xlsx'
	sheet = 'ReutersICEBCTS'
	indexColumn = 0 
	xlsFuturesData =xlExtract(pathToData,sheet,indexColumn) 
	dfFuturesData = xlExtract.extractData(xlsFuturesData, xlsFuturesData.columns,'2017-04-21',  entireTS = True, useLinterpDF = True).dropna()
	print "sheet " + str(sheet) + " extracted"

if power:

	pathToData = 'Data/PowerFutures.xlsx'
	sheets = ['ReutersNordpoolPowerTS_1','ReutersNordpoolPowerTS_2']
	indexColumn = 0
	print "reading sheet  " + str(sheet)

	dfFuturesData = xlExtract.extractData(xlExtract(pathToData,sheet[0],indexColumn), xlsFuturesData.columns,'24-03-2017',  entireTS = True, useLinterpDF = True)
	dfFuturesData = dfFuturesData.join(xlExtract.extractData(xlExtract(pathToData,sheet[1],indexColumn), xlsFuturesData.columns,'24-03-2017',  entireTS = True, useLinterpDF = True))
	storeToHDF5('PowerFutures.hdf5','PowerFutures', dfFuturesData)		
	EONIA = True




