from xlExtract import xlExtract
import pandas as pd
import numpy as np
from h5pyStorage import storeToHDF5, loadFromHDF5
import os.path
from pdb import set_trace
import datetime
from plot_tools import plot_diffs, plot_prices


class futuresInterestPairs(object):

	def __init__(self,dfFuturesData,ForwarRatesdMat,datesInterest):
	
		self.futureNames = dfFuturesData.columns

		for column in dfFuturesData.columns[0:2]: #per future
			datesFutures = dfFuturesData[column].dropna().index #<class 'pandas.tseries.index.DatetimeIndex'
			datesFutures = np.array(datesFutures, dtype = 'datetime64[ns]') #<type 'numpy.ndarray'> of <numpy.datetime64'> elements
			

			self.interestRates = self.getCorrespondingInterestRates(column,datesFutures,ForwarRatesdMat,datesInterest)
			if len(self.interestRates) < 10: #no use in doing this if we have very few matching dates for reinvesting
				print "matching interest rates not found for " + str(len(self._unusedDatesFutures)) + " number of dates"
				print "number of futures dates is " + str(len(datesFutures))
			else:
				self.fut_forStartDays_of_position = self._datesFutures[1:]
				self.fut_forDiffs = []
				self.fut_forEndDay_of_position = self._datesFutures[0]

				endPrice = dfFuturesData[column].loc[self._datesFutures[0]]

				for i in xrange(len(self._datesFutures[1:])): #_datesFutures are those who have a corresponding interest rate position
					
					startPrice = dfFuturesData[column].loc[self._datesFutures[i]]
					forward_long = self.getForwardPosition(startPrice, endPrice)
					futures_long = self.getFuturesPosition(dfFuturesData[column].loc[self._datesFutures[i:-1]],self.interestRates )
					self.fut_forDiffs.append(futures_long - forward_long) #TODO
				#plot_diffs(self.fut_forStartDays_of_position , self.fut_forEndDay_of_position , self.fut_forDiffs, column)
				plot_prices(self._datesFutures, dfFuturesData[column].loc[self._datesFutures], column)
	 #----------USAGE --------

	 
	def getCorrespondingInterestRates(self, column, datesFutures,ForwarRatesdMat,datesInterest):
		interestRates = []
		ForwarRatesdMat_index = []
		self._datesFutures = []
		self._unusedDatesFutures = []
		timeToMats = []
		timeToMat = len(datesFutures)
		print "finding corresponding interest rates for future " + column
		i = 0
		for date in datesFutures:
			i += 1
			if datesInterest[datesInterest['dates'].values == date].index.tolist(): #spara 'ven vilka datum som finns'
				#print datesInterest[datesInterest['dates'].values == date].index.tolist()
				ForwarRatesdMat_index.append(datesInterest[datesInterest['dates'].values == date].index.tolist()[0])

				self._datesFutures.append(date) #only save those who have a corresponding interest rate date
				timeToMats.append(timeToMat - i) 
			else: 
				 self._unusedDatesFutures.append(datesInterest[datesInterest['dates'].values == date].index.tolist())

		return ForwarRatesdMat[ForwarRatesdMat_index,timeToMat]



	def getForwardPosition(self, strikePrice, maturityPrice):
		return  maturityPrice - strikePrice

	def getFuturesPosition(self, futurePrices,interestRates):
		strike = futurePrices.values[0]
		futuresPosition = 0
		for i in xrange(len(futurePrices)):
			futuresPosition += (futurePrices.values[i] - strike)*interestRates[i]
		return futuresPosition


# --------------- start program ---------------

# CHOOSE COMMODITY
GOLD = True
ALU= False
OIL = False
POWER = False


# -----------------------------------------------

def getInterestRateDates(EONIA = True, FFE = False):
	if EONIA:
		print "calculations using EONIA rates"
		pathToData = 'Data/OIS_Data.xlsx'
		sheet = 'EONIA_MID'
		maturitiesInterest = loadFromHDF5('EONIAask.hdf5','times') #maturities, column wise of interest data
		datesInterest = pd.read_excel(pathToData,sheet ,parse_cols = [0]) #dates, row wise of interest rate data
		datesInterest['dates'] = pd.to_datetime(datesInterest['dates'])  # pandas.core.frame.DataFrame of 'numpy.datetime64'> elements
		return datesInterest
	if FFE : #TODO
		print "calculations using FFE rates"
		return []
	else:
		print "no interest rate choosen"

def getInterestRates():
	return loadFromHDF5('EONIAask.hdf5','MATLABForwardMat') #interest rates data, type



if GOLD:
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
		dfFuturesData = xlExtract.extractData(xlsFuturesData, xlsFuturesData.columns,'2017-04-21',  entireTS = True, useLinterpDF = True).dropna(how = 'all')
		print "sheet " + str(sheet) + " extracted"	
		gold_futures_realisation = futuresInterestPairs(dfFuturesData,ForwarRatesdMat,datesInterest)
		

if ALU:
	pass

if OIL:
	pathToData = 'Data/OilFutures.xlsx'
	sheet = 'ReutersICEBCTS'
	indexColumn = 0 
	xlsFuturesData =xlExtract(pathToData,sheet,indexColumn) 
	dfFuturesData = xlExtract.extractData(xlsFuturesData, xlsFuturesData.columns,'2017-04-21',  entireTS = True, useLinterpDF = True).dropna()
	print "sheet " + str(sheet) + " extracted"

if POWER:

	pathToData = 'Data/PowerFutures.xlsx'
	sheets = ['ReutersNordpoolPowerTS_1','ReutersNordpoolPowerTS_2']
	indexColumn = 0
	print "reading sheet  " + str(sheet)

	dfFuturesData = xlExtract.extractData(xlExtract(pathToData,sheet[0],indexColumn), xlsFuturesData.columns,'24-03-2017',  entireTS = True, useLinterpDF = True)
	dfFuturesData = dfFuturesData.join(xlExtract.extractData(xlExtract(pathToData,sheet[1],indexColumn), xlsFuturesData.columns,'24-03-2017',  entireTS = True, useLinterpDF = True))
	storeToHDF5('PowerFutures.hdf5','PowerFutures', dfFuturesData)		
	EONIA = True




