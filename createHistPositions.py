
from __future__ import division
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
ONE_PLOT_PER_FUTURE = False  #this gives a lot of plots, beware
SUMMARY_PLOTS = True

if not ONE_PLOT_PER_FUTURE and not SUMMARY_PLOTS:
	print "plotting is off"



# CHOOSE COMMODITY
GOLD = True
ALU= False
OIL = False
POWER = False

#-------------------------------------------------------------
#
#        Creating futures and forwards position from the data
#
# ---------------------------------------------------------------
class futuresInterestPairs(object):



# ------------- run through the data from excel column per coumn--------------------
	def __init__(self,dfFuturesData,ZeroCouponMat,datesInterest):
		self.futureNames = dfFuturesData.columns
		dfFuturesData = dfFuturesData.reindex(index=dfFuturesData.index[::-1]) #inverse dates and data
		fut_for_diffs = []
		maturities_days = []
		startDates = []
		nameOfFutures = []
		interestRates_at_startDates = []

		#utkommanterat

		for column in dfFuturesData.columns: #per future

			futuresData = dfFuturesData[column].dropna()	

			#make sure dates of futures and interest rates are of the same type
			datesFutures = np.array(futuresData.index, dtype = 'datetime64[ns]')  #<type 'numpy.ndarray'> of <numpy.datetime64'> elements
			#datesInterest = np.array(datesInterest, dtype = 'datetime64[ns]') #<type 'numpy.ndarray'> of <numpy.datetime64'> elements



			self.interestRates = self.getCorrespondingInterestRates(column,datesFutures,ZeroCouponMat,datesInterest)
			
			
			if len(self.interestRates) > 10: #no use in doing this if we have very few matching dates for reinvesting

				if ONE_PLOT_PER_FUTURE:
					self.do_OnePlotPerFuture(futuresData.loc[self._datesFutures], self._datesFutures, self.interestRates, column)
				if SUMMARY_PLOTS:
					interestRates_at_startDates.append(self.interestRates[0])
					nameOfFutures.append(column)
					startPrice = futuresData.loc[self._datesFutures[0]]
					endPrice = futuresData.loc[self._datesFutures[-1]]

					futures_long = self.getFuturesPosition(futuresData.loc[self._datesFutures],self.interestRates)
					forwards_short = -1*self.getForwardPosition(startPrice, endPrice)


					
					if forwards_short != 0: #in power, the price will not have moved at all, yeilding forward = 0
						fut_for_diffs.append((forwards_short + futures_long[-1])/abs(forwards_short))
						#fut_for_diffs.append((forwards_short + futures_long[-1]))
						maturities_days.append(len(futuresData))
						startDates.append(self._datesFutures[0])
						

					else:
						pass
						# print "futures_long = " + str(futures_long[-1])
						# print "forwards_short = " + str(forwards_short)
						# print "column = " + str(column)
						# print "interestRates_at_startDates = " + str(interestRates_at_startDates[-1])
						# print "startPrice = " + str(startPrice)
						# print "endPrice = " + str(endPrice)
						# print "fut_for_diffs = " + str(fut_for_diffs[-1])
						
					
			else:	
				pass			
				# print "matching interest rates not found for " + str(column)
				# print "number of futures dates is " + str(len(self.interestRates))



		if SUMMARY_PLOTS:
			print "summarizing all futures"

			print "len(maturities_days) = "  + str(len(maturities_days))
			print "len(fut_for_diffs) = "  + str(len(fut_for_diffs))
			print "len(startDates) = "  + str(len(startDates))
			print "len(interestRates_at_startDates) = "  + str(len(interestRates_at_startDates))
			self.saveArgsForSummaryPlot(fut_for_diffs,maturities_days, startDates,nameOfFutures,interestRates_at_startDates)
		

# ------------- matching interest rates to futures prices--------------------
	def getCorrespondingInterestRates(self, column, datesFutures,ZeroCouponMat,datesInterest):
		ZeroCouponMat_index = []
		self._datesFutures = []
		_unusedDatesFutures = []
		interestRates =[]
		timeToMats = []
		timeToMat = len(datesFutures)
		i = 0
		for date in datesFutures:
			i += 1
			if datesInterest[datesInterest == date].index.tolist(): #spara 'ven vilka datum som finns'
				#print datesInterest[datesInterest['dates'].values == date].index.tolist()
				ZeroCouponMat_index.append(datesInterest[datesInterest == date].index.tolist()[0])

				self._datesFutures.append(date) #only save those who have a corresponding interest rate date
				timeToMats.append(timeToMat - i) 
			else: 
				 _unusedDatesFutures.append(datesInterest[datesInterest == date].index.tolist())


		return ZeroCouponMat[ZeroCouponMat_index,timeToMats]	#interestRates is now a matrix of all possible spot rates for the correct dates.
		


# ------------- future and forward position--------------------


	def getForwardPosition(self, strikePrice, maturityPrice): #returns one value
		return  maturityPrice - strikePrice


	def getFuturesPosition(self, futurePrices,interestRates): #returns list
		futuresPositions = [0]
		futuresPosition = float(0)
		for i in xrange(1,len(futurePrices)): #change back
			maturity = float(len(futurePrices) - i )/252#

			#reinvest or borrow 
			futuresPosition += float((futurePrices.values[i] -futurePrices.values[i-1]))*float(np.exp(float(interestRates[i]*maturity)))

			#interest rate = zero
			#futuresPosition += (futurePrices.values[i] -futurePrices.values[i-1])*np.exp(0*maturity)
			futuresPositions.append(futuresPosition)
		return futuresPositions

# -------------------------------------------------------------
#
#        Do  one plot per future 
#
# ---------------------------------------------------------------
	def do_OnePlotPerFuture(self,futuresPrices, _datesFutures, interestRates, column):
		
		endPrice = futuresPrices.loc[_datesFutures[-1]]
	 	startPrice = futuresPrices.loc[self._datesFutures[0]]
		#how much the futures position is worth over time when you buy at startti
		futures_pos_over_time = self.getFuturesPosition(futuresPrices, interestRates) #list
		# futures_value_over_time = self.getFuturesValue()
		# forward_value_over_time = elf.getForwardValue()

	 	forward_pos_over_time = [self.getForwardPosition(startPrice, endPrice)] *len(futures_pos_over_time)	
 		forward_long = forward_pos_over_time
 		futures_long = futures_pos_over_time

 		fut_forDiffs = [fut_i - for_i for fut_i, for_i in zip(futures_long, forward_long)]
		#startDates, diffs, prices, nameOfFuture
		_onePlotPerFuture(_datesFutures,fut_forDiffs, futures_pos_over_time, futuresPrices ,forward_pos_over_time, interestRates, column)  
		
	# -------------for usage of arguments aoutside of class -------------------

	def saveArgsForSummaryPlot(self, fut_for_diffs,maturities_days, startDates,nameOfFutures,interestRates_at_startDates):
		self.fut_for_diffs = fut_for_diffs
		self.maturities_days = maturities_days
		self.startDates = startDates
		self.columns = nameOfFutures
		self.interestRates_at_startDates = interestRates_at_startDates

# -------------------------------------------------------------
#
#         Summary plots
#
# ---------------------------------------------------------------

def summaryPlot(futures_realisation):
	fut_for_diffs = []
	maturities_days = [] 
	startDates = []
	columns = []
	interestRates_at_startDates = []

	print "len(futures_realisation) = "  + str(len(futures_realisation))

	for i in xrange(len(futures_realisation)):
		#print "len(futures_realisation[i].fut_for_diffs) = " + str(len(futures_realisation[i].fut_for_diffs))
		fut_for_diffs.extend(futures_realisation[i].fut_for_diffs)
		maturities_days.extend(futures_realisation[i].maturities_days)
		startDates.extend(futures_realisation[i].startDates)
		columns.extend(futures_realisation[i].columns)
		interestRates_at_startDates.extend(futures_realisation[i].interestRates_at_startDates)

	print "len(maturities_days) = "  + str(len(maturities_days))
	print "len(fut_for_diffs) = "  + str(len(fut_for_diffs))
	print "len(startDates) = "  + str(len(startDates))
	print "len(interestRates_at_startDates) = "  + str(len(interestRates_at_startDates))

	maturities_days, fut_for_diffs, startDates, interestRates_at_startDates = sortDataOnMaturityDays(maturities_days, fut_for_diffs, startDates, interestRates_at_startDates)		

#	print "printing maturities in days"
#	print maturities_days
	
	if POWER:
		titleStrings = ["Diffs in Power position: \n maturities uo to 3M", "Diffs in Power position: \n maturities uo to 3M-6M", "Diffs in Power position: \n short for + long future - maturities 6M - 1Y", "Diffs in Power position: \n short for + long future - maturities 1Y - 2Y", "Diffs in Power position: \n short for + long future - maturities 2Y - 3Y" , "Diffs in Power position: \n short for + long future - maturities 3Y - 4Y", "Diffs in Power position: \n short for + long future - maturities 4Y - 7Y"   ]
		matGroups = [718, 1114, 1331, 1405, 1508, 1526, 1544]
		savePath = "Plots/Diffs/Power/diffs_matGroup" 
	if OIL:
		titleStrings = ["Diffs in Oil position: \n maturities 1M - 3M", "Diffs in Oil position: \n short for + long future - maturities 476 - 485 days", "Diffs in Oil position: \n short for + long future - maturities 1247 - 1513 days" ]
		matGroups = [80, 133, 160]
		#get corresponding interest date for the maturities for the specific startdate
		savePath = "Plots/Diffs/OIL/diffs_matGroup"
	if GOLD:
		#multiple plots over the different maturity groups
		titleStrings = ["Diffs in Gold position: \n maturities 1M- 2M", "Diffs in Gold position: \n short for + long future - maturities 2Y - 3Y", "Diffs in Gold position: \n short for + long future - maturities 3Y - 6Y" , "Diffs in Gold position: \n maturities 1M- 6Y"]
		savePath = "Plots/Diffs/GOLD/diffs_matGroup" 
		matGroups = [222, 398, 443]
		#get corresponding interest date for the maturities for the specific startdates


	#do the plots
	i  = 0
	for mat_index, titleString in zip(matGroups, titleStrings):
		
		saveString = savePath + str(mat_index) + ".png"
		_summaryPlot(titleString, saveString, startDates[i:mat_index], fut_for_diffs[i:mat_index], maturities_days[i:mat_index], interestRates_at_startDates[i:mat_index])
		#print maturities_days[i:mat_index]
		i = mat_index
	titleString = titleStrings[-1]
	saveString = savePath + "all_mats.png"
	_summaryPlot(titleString, saveString, startDates, fut_for_diffs, maturities_days, interestRates_at_startDates)





# -------------------------------------------------------------
#
#        Modify data for plotting
#
# ---------------------------------------------------------------

def sortDataOnMaturityDays( maturities_days, fut_for_diffs, startDates, interestRates_at_startDates):
	#sort the maturities and fut_for_diffs accordingly, unzip
	return zip(*sorted(zip(maturities_days,fut_for_diffs, startDates, interestRates_at_startDates)))










# -------------------------------------------------------------
#
#        EXTRACT DATA functions for extracing rates
#
# ---------------------------------------------------------------

def getInterestRateDates(EONIA, FFE, USGG):
	if EONIA:
		print "calculations using EONIA rates"
		pathToData = 'Data/OIS_Data.xlsx'
		sheet = 'EONIA_MID'
		#maturitiesInterest = loadFromHDF5('EONIAask.hdf5','times') #maturities, column wise of interest data
		datesInterest = pd.read_excel(pathToData,sheet ,parse_cols = [0]) #dates, row wise of interest rate data
		datesInterest['dates'] = pd.to_datetime(datesInterest['dates'])  # pandas.core.frame.DataFrame of 'numpy.datetime64'> elements
		return datesInterest
	if FFE :
		print "calculations using FFE rates"
		pathToData = 'Data/OIS_Data.xlsx'
		sheet = 'FFE_MID'
		#maturitiesInterest = loadFromHDF5('EONIAask.hdf5','times') #maturities, column wise of interest data
		datesInterest = pd.read_excel(pathToData,sheet ,parse_cols = [0]) #dates, row wise of interest rate data
		datesInterest['dates'] = pd.to_datetime(datesInterest['dates'])  # pandas.core.frame.DataFrame of 'numpy.datetime64'> elements
		return datesInterest
	if USGG :
		print "calculations using USSG rates"
		datesInterest = getInterestRates(EONIA = False, FFE = False, USGG = True).index 


		return datesInterest


	else:
		print "no interest rate choosen"

def getInterestRates(EONIA, FFE, USGG):
	if EONIA:
		return loadFromHDF5('EONIAask.hdf5','ZCMat') #interest rates data, type
	if FFE:
		print "type(loadFromHDF5('FFEmid.hdf5','ZCMat'))" + str(type(loadFromHDF5('FFEmid.hdf5','ZCMat')))
		return loadFromHDF5('FFEmid.hdf5','ZCMat')

	if USGG:
		dataCutoff = 4100
		OISsheet = 'USGG_MID'
		pathToData = 'Data/OIS_data.xlsx'

		ZCData = xlExtract(pathToData, OISsheet, 0) # Load from data frame to get indexes and columns
		dfZCData = ZCData.dflinterp[:dataCutoff]

		#FFE rates added to first month as USSG rates does not exist for shorter times
		originalZCMat = loadFromHDF5('USGGmid.hdf5','ZCMat')
		originalTimes = loadFromHDF5('USGGmid.hdf5','times')
		print originalTimes[0]

		# Extend to include all times down to 1 day
		extraTimes = np.arange(1/365,originalTimes[0]-1/365,1/365)
		times = np.append(extraTimes, originalTimes)

		extraSteps = extraTimes.shape[0]
		extendedZCMat = np.repeat(originalZCMat[:,0:1],extraSteps, axis=1)
		ZCMat = np.column_stack((extendedZCMat, originalZCMat))

		dfZCMat = pd.DataFrame(data=ZCMat[:dataCutoff,:], index=ZCData.index[:dataCutoff], columns=times) # Dataframe of ZC matrix to use for date-matching


		return dfZCMat.values #pandas dataframe


# -------------------------------------------------------------
#
#                EXTRACT DATA main code
#
# ---------------------------------------------------------------

if GOLD:
	#interest rate dates and data
	
	ZeroCouponMat = getInterestRates(EONIA = False, FFE = False, USGG = True)
	datesInterest = getInterestRateDates(EONIA = False, FFE = True, USGG = True)
	if len(ZeroCouponMat) != len(datesInterest): #if they are not the same length, bad data has been cut of
		datesInterest = datesInterest.iloc[0:len(ZeroCouponMat)]

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
		gold_futures_realisation.append( futuresInterestPairs(dfFuturesData,ZeroCouponMat,datesInterest))


	if SUMMARY_PLOTS:
		summaryPlot(gold_futures_realisation)

if ALU:
	#interest rate dates and data
	#datesInterest = getInterestRateDates(EONIA = False, FFE = True)
	#ZeroCouponMat = getInterestRates()
	pass

if OIL:
	#interest rate dates and data
	ZeroCouponMat = getInterestRates(EONIA = False, FFE = False, USGG = True)
	datesInterest = getInterestRateDates(EONIA = False, FFE = True, USGG = True)
	if len(ZeroCouponMat) != len(datesInterest): #if they are not the same length, bad data has been cut of
		datesInterest = datesInterest.iloc[0:len(ZeroCouponMat)]


	pathToData = 'Data/OilFutures.xlsx'
	sheet = 'ReutersICEBCTS'
	indexColumn = 0 
	xlsFuturesData =xlExtract(pathToData,sheet,indexColumn) 
	dfFuturesData = xlExtract.extractData(xlsFuturesData, xlsFuturesData.columns,'2017-04-21',  entireTS = True, useLinterpDF = False).dropna()
	
	oil_futures_realisation = futuresInterestPairs(dfFuturesData,ZeroCouponMat,datesInterest)
	
	if SUMMARY_PLOTS:
		summaryPlot(oil_futures_realisation)

if POWER:
	#interest rate dates and data
	datesInterest = getInterestRateDates(EONIA = True, FFE = False, USGG = False)
	ZeroCouponMat = getInterestRates(EONIA = True, FFE = False, USGG = False)
	if len(ZeroCouponMat) != len(datesInterest): #if they are not the same length, bad data has been cut of
		datesInterest = datesInterest.iloc[0:len(ZeroCouponMat)]

	pathToData = 'Data/PowerFutures.xlsx'
	sheets = ['ReutersNordpoolPowerTS_1','ReutersNordpoolPowerTS_2']
	power_futures_realisation = []
	indexColumn = 0
	for sheet in sheets:
		xlsFuturesData = xlExtract(pathToData,sheet,indexColumn)
		dfFuturesData = xlExtract.extractData(xlsFuturesData, xlsFuturesData.columns,'24-03-2017',  entireTS = True, useLinterpDF = False).dropna(how = 'all')
		print "sheet " + str(sheet) + " extracted"	
		power_futures_realisation.append( futuresInterestPairs(dfFuturesData,ZeroCouponMat,datesInterest))
		
	if SUMMARY_PLOTS:
		summaryPlot(power_futures_realisation)






# ## -------------one plot per future --------------------
# 	def do_OnePlotPerFuture(self,futuresData, _datesFutures, interestRates, column):
# 		#how much the futures position is worth over time when you buy at starttime
# 		futures_pos_over_time = self.getFuturesPosition(futuresData.loc[_datesFutures], interestRates) #list
# 		futures_value = self.getFuturesValue()

# 		#the result when you enter the positions at day 1, day 2 etc
# 	 	endPrice = futuresData.loc[_datesFutures[-1]]

# 	 	#create a plot of the forward value
# 	 	forward_pos_over_time = [self.getForwardPosition(futuresData.loc[self._datesFutures[0]], endPrice)] *len(futures_pos_over_time)
# 	 	fut_forDiffs = []
# 	 	for i in xrange(len(_datesFutures)): #_datesFutures are those who have a corresponding interest rate position


# 	 		startPrice = futuresData.loc[self._datesFutures[i]]
# 	 		forward_long = self.getForwardPosition(startPrice, endPrice)
# 	 		futures_long = self.getFuturesPosition(futuresData.loc[_datesFutures[i:]], interestRates)
# 	 		fut_forDiffs.append(futures_long[-1] - forward_long) 


		
# 		#startDates, diffs, prices, nameOfFuture
# 		_onePlotPerFuture(_datesFutures,fut_forDiffs, futures_pos_over_time, forward_pos_over_time, futuresData.loc[_datesFutures], column)  
		






