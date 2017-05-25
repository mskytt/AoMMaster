from xlExtract import xlExtract
import pandas as pd
import numpy as np
from h5pyStorage import storeToHDF5, loadFromHDF5
import os.path
from pdb import set_trace
import datetime
from plot_tools import onePlotPerFuture, summaryPlot


# --------------- start program ---------------

#define what you want to plot 
ONE_PLOT_PER_FUTURE = True #this gives a lot of plots, beware
SUMMARY_PLOTS = False

if not ONE_PLOT_PER_FUTURE and not SUMMARY_PLOTS:
	print "plotting is off"



# CHOOSE COMMODITY
GOLD = True
ALU= False
OIL = False
POWER = False

# --------------- doing things with the data---------------

class futuresInterestPairs(object):

	def __init__(self,dfFuturesData,ForwarRatesdMat,datesInterest):
		self.futureNames = dfFuturesData.columns
		dfFuturesData = dfFuturesData.reindex(index=dfFuturesData.index[::-1]) #inverse dates and data
		fut_for_diffs = []
		maturities_days = []
		startDates = []


		for column in dfFuturesData.columns[0:2]: #per future

			#print "on  " + column 
			datesFutures = np.array(dfFuturesData[column].dropna().index, dtype = 'datetime64[ns]')  #<type 'numpy.ndarray'> of <numpy.datetime64'> elements
			self.interestRates = self.getCorrespondingInterestRates(column,datesFutures,ForwarRatesdMat,datesInterest)


			if len(self.interestRates) < 10: #no use in doing this if we have very few matching dates for reinvesting
				print "matching interest rates not found for " + str(column)
				print "number of futures dates is " + str(len(datesFutures))
			else:
				if ONE_PLOT_PER_FUTURE:
					self.do_OnePlotPerFuture(dfFuturesData, self._datesFutures, self.interestRates,column)

				startPrice = dfFuturesData[column].loc[self._datesFutures[0]]
				endPrice = dfFuturesData[column].loc[self._datesFutures[-1]]

				futures_long = self.getFuturesPosition(dfFuturesData[column].loc[self._datesFutures],self.interestRates)
				forwards_short = -1*self.getForwardPosition(startPrice, endPrice)
				fut_for_diffs.append(forwards_short + futures_long)
				maturities_days.append(len(dfFuturesData[column]))
				startDates.append(self._datesFutures[0])

		if SUMMARY_PLOTS:
			self.do_OneSummaryPlot(fut_for_diffs,maturities_days, startDates)
	

				
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



	def do_OnePlotPerFuture(self,dfFuturesData, _datesFutures, interestRates, column):
		#how much the futures position is worth over time
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
		onePlotPerFuture(_datesFutures,fut_forDiffs, futures_pos_over_time, dfFuturesData[column].loc[_datesFutures], column)  
		



	def do_OneSummaryPlot(self,fut_for_diffs,prices, maturities_days, startDates): 

			#self.surfPlot_diffs(fut_for_diffs,maturities_days, startDates			self._2dPlot_diffs(fut_for_diffs,maturities_days, startDates)






	def surfPlot_diffs(self, fut_for_diffs,maturities_days, startDates):
		#sort on maturities?

		matrix = [[]]
		surfPlot(matrix, startDates )

	def _2dPlot_diffs(self,fut_for_diffs,maturities_days, startDates):
		plot_diffs_sameStart(startDates,fut_for_diffs)
		plot_diffs_mat(fut_for_diffs,maturities_days)


	def getForwardPosition(self, strikePrice, maturityPrice): #returns one value
		return  maturityPrice - strikePrice

	def getFuturesPosition(self, futurePrices,interestRates): #returns list
		futuresPosition = [0]
		for i in xrange(1,len(futurePrices)):
			futuresPosition.append((futuresPosition[i-1] + futurePrices.values[i] -futurePrices.values[i-1])*np.exp(interestRates[i]))

		return futuresPosition




# ------------------Getting data-----------------------------

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




	# 	#plot_diffs_in_same(self._datesFutures ,self._datesFutures[-1] , fut_forDiffs, column)
	# 	plot_value(_datesFutures,futuresPositions_movingStartDates, column) #moving startDate
	# 	plot_diffs(_datesFutures ,_datesFutures[-1] , fut_forDiffs, column)
	# 	plot_prices(_datesFutures, dfFuturesData[column].loc[_datesFutures], column)

