from xlExtract import xlExtract
import pandas as pd
import numpy as np
from h5pyStorage import storeToHDF5, loadFromHDF5
import os.path
from pdb import set_trace
import datetime
from plot_tools import _onePlotPerFuture, _summaryPlot


# --------------- start program ---------------

#define what you want to plot 
ONE_PLOT_PER_FUTURE = False #this gives a lot of plots, beware
SUMMARY_PLOTS = True

if not ONE_PLOT_PER_FUTURE and not SUMMARY_PLOTS:
	print "plotting is off"



# CHOOSE COMMODITY
GOLD = True
ALU= False
OIL = False
POWER = False

# --------------- doing things with the data---------------

class futuresInterestPairs(object):



# ------------- run through the data from excel column per coumn--------------------
	def __init__(self,dfFuturesData,ForwarRatesdMat,datesInterest):
		self.futureNames = dfFuturesData.columns
		dfFuturesData = dfFuturesData.reindex(index=dfFuturesData.index[::-1]) #inverse dates and data
		fut_for_diffs = []
		maturities_days = []
		startDates = []
		nameOfFutures = []


		for column in dfFuturesData.columns: #per future

			#print "on  " + column 
			futuresData = dfFuturesData[column].dropna()
			datesFutures = np.array(futuresData.index, dtype = 'datetime64[ns]')  #<type 'numpy.ndarray'> of <numpy.datetime64'> elements

			self.interestRates = self.getCorrespondingInterestRates(column,datesFutures,ForwarRatesdMat,datesInterest)

			if len(self.interestRates) > 10: #no use in doing this if we have very few matching dates for reinvesting
				# print "matching interest rates not found for " + str(column)
				# print "number of futures dates is " + str(len(datesFutures))
	
				if ONE_PLOT_PER_FUTURE:
					self.do_OnePlotPerFuture(futuresData, self._datesFutures, self.interestRates,column)
				else:
					nameOfFutures.append(column)
					startPrice = futuresData.loc[self._datesFutures[0]]
					endPrice = futuresData.loc[self._datesFutures[-1]]

					futures_long = self.getFuturesPosition(futuresData.loc[self._datesFutures],self.interestRates)
					forwards_short = -1*self.getForwardPosition(startPrice, endPrice)
					fut_for_diffs.append(forwards_short + futures_long[-1])
					maturities_days.append(len(futuresData))
					startDates.append(self._datesFutures[0])

		if SUMMARY_PLOTS:
			self.saveArgsForSummaryPlot(fut_for_diffs,maturities_days, startDates,nameOfFutures)
	

# ------------- matching interest rates to futures prices--------------------
	def getCorrespondingInterestRates(self, column, datesFutures,ForwarRatesMat,datesInterest):
		ForwarRatesdMat_index = []
		self._datesFutures = []
		_unusedDatesFutures = []
		interestRates =[]
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


		return ForwarRatesMat[ForwarRatesdMat_index,timeToMats]	#interestRates is now a matrix of all possible spot rates for the correct dates.
		



# ------------- future and forward position--------------------


	def getForwardPosition(self, strikePrice, maturityPrice): #returns one value
		return  maturityPrice - strikePrice


	def getFuturesPosition(self, futurePrices,interestRates): #returns list
		futuresPositions = [0]
		futuresPosition = 0
		for i in xrange(1,len(futurePrices)): #change back
			maturity = (float(len(futurePrices) - i))/365
			futuresPosition += (futurePrices.values[i] -futurePrices.values[i-1])*np.exp(interestRates[i]*maturity)
			futuresPositions.append(futuresPosition)
		return futuresPositions


# -------------one plot per future --------------------
	def do_OnePlotPerFuture(self,dfFuturesData, _datesFutures, interestRates, column):
		#how much the futures position is worth over time when you buy at starttime
		futures_pos_over_time = self.getFuturesPosition(dfFuturesData[column].loc[_datesFutures], interestRates) #list

		#the result when you enter the positions at day 1, day 2 etc
	 	endPrice = dfFuturesData[column].loc[_datesFutures[-1]]
	 	fut_forDiffs = []
	 	for i in xrange(len(_datesFutures)): #_datesFutures are those who have a corresponding interest rate position

	 		startPrice = dfFuturesData[column].loc[self._datesFutures[i]]
	 		forward_long = self.getForwardPosition(startPrice, endPrice)
	 		futures_long = self.getFuturesPosition(dfFuturesData[column].loc[_datesFutures[i:]], interestRates)
	 		fut_forDiffs.append(futures_long[-1] - forward_long) 

		#startDates, diffs, prices, nameOfFuture
		_onePlotPerFuture(_datesFutures,fut_forDiffs, futures_pos_over_time, dfFuturesData[column].loc[_datesFutures], column)  
		
	# -------------for usage of arguments aoutside of class -------------------

	def saveArgsForSummaryPlot(self,fut_for_diffs, maturities_days, startDates, columns):
		self.fut_for_diffs = fut_for_diffs
		self.maturities_days = maturities_days
		self.startDates = startDates
		self.columns = columns


# ------------------Plotting summaries-----------------------------

def summaryPlot(futures_realisation):
	print futures_realisation
	fut_for_diffs = []
	maturities_days = [] 
	startDates = []
	columns = []


	for i in xrange(len(futures_realisation)):
		print "len(futures_realisation[i].fut_for_diffs) = " + str(len(futures_realisation[i].fut_for_diffs))
		fut_for_diffs.append(futures_realisation[i].fut_for_diffs)
		maturities_days.append(futures_realisation[i].maturities_days)
		startDates.append(futures_realisation[i].startDates)
		columns.append(futures_realisation[i].columns)




	doSummaryPlot(fut_for_diffs, maturities_days, startDates, columns)



def doSummaryPlot(fut_for_diffs, maturities_days, startDates, columns): 
	#sort the maturities and fut_for_diffs accordingly, unzip
	maturities_days, fut_for_diffs = zip(*sorted(zip(maturities_days,fut_for_diffs)))
	print len(fut_for_diffs) == len(maturities_days)
	_summaryPlot(startDates, fut_for_diffs, maturities_days, columns)
		



# ------------------For getting interest rate data-----------------------------

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


# ------------------Getting data-----------------------------

if GOLD:
	#interest rate dates and data
	datesInterest = getInterestRateDates(EONIA = True, FFE = False)
	ForwarRatesdMat = getInterestRates()

	#futures dates and data
	pathToData =  'Data/GoldFutures.xlsx'
	sheets = ['ReutersCOMEXGoldTS1', 'ReutersCOMEXGoldTS2', 'ReutersCOMEXGoldTS3']
	indexColumn = 0
	dfFuturesData = pd.DataFrame()
	gold_futures_realisation = []
	for sheet in sheets:
		xlsFuturesData = xlExtract(pathToData,sheet,indexColumn)
		dfFuturesData = xlExtract.extractData(xlsFuturesData, xlsFuturesData.columns,'2017-04-21',  entireTS = True, useLinterpDF = False).dropna(how = 'all')
		print "sheet " + str(sheet) + " extracted"	
		gold_futures_realisation.append( futuresInterestPairs(dfFuturesData,ForwarRatesdMat,datesInterest))
	


	summaryPlot(gold_futures_realisation)

if ALU:
	pass

if OIL:
	#interest rate dates and data
	datesInterest = getInterestRateDates(EONIA = True, FFE = False)
	ForwarRatesdMat = getInterestRates()


	pathToData = 'Data/OilFutures.xlsx'
	sheet = 'ReutersICEBCTS'
	indexColumn = 0 
	xlsFuturesData =xlExtract(pathToData,sheet,indexColumn) 
	dfFuturesData = xlExtract.extractData(xlsFuturesData, xlsFuturesData.columns,'2017-04-21',  entireTS = True, useLinterpDF = False).dropna()
	
	oil_futures_realisation = futuresInterestPairs(dfFuturesData,ForwarRatesdMat,datesInterest)

	summaryPlot(oil_futures_realisation)

if POWER:

	#interest rate dates and data
	datesInterest = getInterestRateDates(EONIA = True, FFE = False)
	ForwarRatesdMat = getInterestRates()

	pathToData = 'Data/PowerFutures.xlsx'
	sheets = ['ReutersNordpoolPowerTS_1','ReutersNordpoolPowerTS_2']
	power_futures_realisation = []
	indexColumn = 0
	for sheet in sheets:
		xlsFuturesData = xlExtract(pathToData,sheet,indexColumn)
		dfFuturesData = xlExtract.extractData(xlsFuturesData, xlsFuturesData.columns,'24-03-2017',  entireTS = True, useLinterpDF = False).dropna(how = 'all')
		print "sheet " + str(sheet) + " extracted"	
		power_futures_realisation.append( futuresInterestPairs(dfFuturesData,ForwarRatesdMat,datesInterest))

	summaryPlot(power_futures_realisation)



	EONIA = True








