from xlExtract import xlExtract
import pandas as pd
import numpy as np
from h5pyStorage import storeToHDF5, loadFromHDF5
import os.path
from pdb import set_trace
import datetime
from plot_tools import plot_diffs, plot_prices, plot_value, plot_diffs_in_same, surfPlot,plot_diffs_sameStart,plot_diffs_mat


class futuresInterestPairs(object):

	def __init__(self,dfFuturesData,ForwarRatesdMat,datesInterest):
		self.futureNames = dfFuturesData.columns
		dfFuturesData = dfFuturesData.reindex(index=dfFuturesData.index[::-1]) #inverse dates and data
		fut_for_diffs = []
		maturities_days = []
		startDates = []


		for column in dfFuturesData.columns[0:1]: #per future

			#print "on  " + column 
			datesFutures = np.array(dfFuturesData[column].dropna().index, dtype = 'datetime64[ns]')  #<type 'numpy.ndarray'> of <numpy.datetime64'> elements
			self.interestRates = self.getCorrespondingInterestRates(column,datesFutures,ForwarRatesdMat,datesInterest)


			if len(self.interestRates) < 10: #no use in doing this if we have very few matching dates for reinvesting
				print "matching interest rates not found for " + str(column)
				print "number of futures dates is " + str(len(datesFutures))
			else:
				self.plot_RealisedDiffs_movingStart(dfFuturesData,column)
				startPrice = dfFuturesData[column].loc[self._datesFutures[0]]
				endPrice = dfFuturesData[column].loc[self._datesFutures[-1]]

				futures_long = self.getFuturesPosition(dfFuturesData[column].loc[self._datesFutures],self.interestRates)
				forwards_short = -1*self.getForwardPosition(startPrice, endPrice)
				fut_for_diffs.append(forwards_short + futures_long)
				maturities_days.append(len(dfFuturesData[column]))
				startDates.append(self._datesFutures[0])
		#self.surfPlot_diffs(fut_for_diffs,maturities_days, startDates)
		self._2dPlot_diffs(fut_for_diffs,maturities_days, startDates)



				
	#----------USAGE --------
	def getCorrespondingInterestRates(self, column, datesFutures,ForwarRatesMat,datesInterest):
		interestRates = []
		ForwarRatesdMat_index = []
		self._datesFutures = []
		_unusedDatesFutures = []
		timeToMats = []
		timeToMat = len(datesFutures)
		i = 0
		for date in datesFutures:
			i += 1
			if datesInterest[datesInterest['dates'].values == date].index.tolist(): #spara 'ven vilka datum som finns'
				#print datesInterest[datesInterest['dates'].values == date].index.tolist()
				ForwarRatesdMat_index.append(datesInterest[datesInterest['dates'].values == date].index.tolist()[0])

				self._datesFutures.append(date) #only save those who have a corresponding interest rate date
				timeToMats.append(timeToMat - i) 
			else: 
				 _unusedDatesFutures.append(datesInterest[datesInterest['dates'].values == date].index.tolist())
		#print "found corresponding interest rates for future " + column
		return ForwarRatesMat[ForwarRatesdMat_index,timeToMat]



	def plot_RealisedDiffs_movingStart(self,dfFuturesData,column):
		fut_forDiffs = []
		futuresPositions = []
		endPrice = dfFuturesData[column].loc[self._datesFutures[-1]]
		for i in xrange(len(self._datesFutures)): #_datesFutures are those who have a corresponding interest rate position

			startPrice = dfFuturesData[column].loc[self._datesFutures[i]]
			forward_long = self.getForwardPosition(startPrice, endPrice)
			futures_long = self.getFuturesPosition(dfFuturesData[column].loc[self._datesFutures[i:]],self.interestRates)
			fut_forDiffs.append(futures_long - forward_long) #TODO
			futuresPositions.append(futures_long)

		#plot_diffs_in_same(self._datesFutures ,self._datesFutures[-1] , fut_forDiffs, column)
		plot_value(self._datesFutures,futuresPositions, column)
		plot_diffs(self._datesFutures ,self._datesFutures[-1] , fut_forDiffs, column)
		plot_prices(self._datesFutures, dfFuturesData[column].loc[self._datesFutures], column)

	def surfPlot_diffs(self, fut_for_diffs,maturities_days, startDates):
		#sort on maturities?

		matrix = [[]]
		surfPlot(matrix, startDates )

	def _2dPlot_diffs(self,fut_for_diffs,maturities_days, startDates):
		plot_diffs_sameStart(startDates,fut_for_diffs)
		plot_diffs_mat(fut_for_diffs,maturities_days)


	def getForwardPosition(self, strikePrice, maturityPrice):
		return  maturityPrice - strikePrice

	def getFuturesPosition(self, futurePrices,interestRates):
		futuresPosition = 0
		for i in xrange(1,len(futurePrices)):
			futuresPosition += (futurePrices.values[i] -futurePrices.values[i-1])*np.exp(interestRates[i])

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
	return loadFromHDF5('EONIAask.hdf5','ZCMat') #interest rates data, type



if GOLD:
	#interest rate dates and data
	datesInterest = getInterestRateDates(EONIA = True, FFE = False)
	ForwarRatesdMat = getInterestRates()

	#futures dates and data
	pathToData =  'Data/GoldFutures.xlsx'
	sheets = ['ReutersCOMEXGoldTS1', 'ReutersCOMEXGoldTS2', 'ReutersCOMEXGoldTS3']
	indexColumn = 0
	dfFuturesData = pd.DataFrame()
	#for sheet in sheets[0:1]:
	sheet = sheets[0]
	xlsFuturesData = xlExtract(pathToData,sheet,indexColumn)
	dfFuturesData = xlExtract.extractData(xlsFuturesData, xlsFuturesData.columns,'2017-04-21',  entireTS = True, useLinterpDF = False).dropna(how = 'all')
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




