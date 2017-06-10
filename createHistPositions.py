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
			datesFutures = np.array(futuresData.index, dtype = 'datetime64[ns]')  #<type 'numpy.ndarray'> of <numpy.datetime64'> elements

			self.interestRates = self.getCorrespondingInterestRates(column,datesFutures,ZeroCouponMat,datesInterest)
			
			
			if len(self.interestRates) > 10: #no use in doing this if we have very few matching dates for reinvesting


				if ONE_PLOT_PER_FUTURE:
					self.do_OnePlotPerFuture(futuresData.loc[self._datesFutures], self._datesFutures, self.interestRates, column)
				else:
					interestRates_at_startDates.append(self.interestRates[0])
					nameOfFutures.append(column)
					startPrice = futuresData.loc[self._datesFutures[0]]
					endPrice = futuresData.loc[self._datesFutures[-1]]

					futures_long = self.getFuturesPosition(futuresData.loc[self._datesFutures],self.interestRates)
					forwards_short = -1*self.getForwardPosition(startPrice, endPrice)
					fut_for_diffs.append((forwards_short + futures_long[-1])/abs(forwards_short))
					maturities_days.append(len(futuresData))
					startDates.append(self._datesFutures[0])
			else:
				pass
				#print "matching interest rates not found for " + str(column)
				#print "number of futures dates is " + str(len(self.interestRates))



		if SUMMARY_PLOTS:
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
			if datesInterest[datesInterest['dates'].values == date].index.tolist(): #spara 'ven vilka datum som finns'
				#print datesInterest[datesInterest['dates'].values == date].index.tolist()
				ZeroCouponMat_index.append(datesInterest[datesInterest['dates'].values == date].index.tolist()[0])

				self._datesFutures.append(date) #only save those who have a corresponding interest rate date
				timeToMats.append(timeToMat - i) 
			else: 
				 _unusedDatesFutures.append(datesInterest[datesInterest['dates'].values == date].index.tolist())


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


# -------------one plot per future --------------------
	def do_OnePlotPerFuture(self,futuresPrices, _datesFutures, interestRates, column):
		
		endPrice = futuresPrices.loc[_datesFutures[-1]]
		print "endPrice = " + str(endPrice)
	 	startPrice = futuresPrices.loc[self._datesFutures[0]]
	 	print "startPrice = " + str(startPrice)
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


# ------------------Plotting summaries-----------------------------

def summaryPlot(futures_realisation):
	fut_for_diffs = []
	maturities_days = [] 
	startDates = []
	columns = []
	interestRates_at_startDates = []



	for i in xrange(len(futures_realisation)):
		#print "len(futures_realisation[i].fut_for_diffs) = " + str(len(futures_realisation[i].fut_for_diffs))
		fut_for_diffs.extend(futures_realisation[i].fut_for_diffs)
		maturities_days.extend(futures_realisation[i].maturities_days)
		startDates.extend(futures_realisation[i].startDates)
		columns.extend(futures_realisation[i].columns)
		interestRates_at_startDates.extend(futures_realisation[i].interestRates_at_startDates)


	

	maturities_days, fut_for_diffs, startDates, interestRates_at_startDates = sortDataOnMaturityDays(maturities_days, fut_for_diffs, startDates, interestRates_at_startDates)		


	print "printing maturities in days"
	print len(maturities_days)
	print maturities_days

	print "Printing diffs vs startdates in different maturity groups"
	
	if POWER:
		titleStrings = ["Diffs in Power position: \n maturities 57 - 64 days", "Diffs in Power position: \n short for + long future - maturities 476 - 485 days", "Diffs in Power position: \n short for + long future - maturities 1247 - 1513 days" ]
		matGroups = [80, 133, 160]
		savePath = "Plots/Diffs/Power/diffs_matGroup" 
		


	if OIL:
		titleStrings = ["Diffs in Oil position: \n maturities 57 - 64 days", "Diffs in Oil position: \n short for + long future - maturities 476 - 485 days", "Diffs in Oil position: \n short for + long future - maturities 1247 - 1513 days" ]
		matGroups = [80, 133, 160]
		#get corresponding interest date for the maturities for the specific startdate
		savePath = "Plots/Diffs/OIL/diffs_matGroup"
	

	if GOLD:
		#multiple plots over the different maturity groups
		titleStrings = ["Diffs in Gold position: \n maturities 57 - 64 days", "Diffs in Gold position: \n short for + long future - maturities 476 - 485 days", "Diffs in Gold position: \n short for + long future - maturities 1247 - 1513 days" , "Diffs in Gold position: \n maturities 57 - 1513 days"]
		savePath = "Plots/Diffs/GOLD/diffs_matGroup" 
		matGroups = [80, 133, 160]
		#get corresponding interest date for the maturities for the specific startdates


	#do the plot
	i  = 0
	for mat_index, titleString in zip(matGroups, titleStrings):

		saveString = savePath + str(mat_index) + ".png"
		_summaryPlot(titleString, saveString, startDates[i:mat_index], fut_for_diffs[i:mat_index], maturities_days[i:mat_index], interestRates_at_startDates[i:mat_index])
		#print maturities_days[i:mat_index]
		i = mat_index
	titleString = titleStrings[-1]
	saveString = savePath + "all_mats.png"
	_summaryPlot(titleString, saveString, startDates, fut_for_diffs, maturities_days, interestRates_at_startDates)









def sortDataOnMaturityDays( maturities_days, fut_for_diffs, startDates, interestRates_at_startDates):
	#sort the maturities and fut_for_diffs accordingly, unzip
	return zip(*sorted(zip(maturities_days,fut_for_diffs, startDates, interestRates_at_startDates)))




		
def takeMeanOfAllEqualMaturitiesDiffs(maturities_days, fut_for_diffs):


	#convert arrays into np arrays in order to use specific np function
	maturities_days_temp = np.asarray(maturities_days)
	fut_for_diffs_temp = np.asarray(fut_for_diffs)


	maturities_days = []
	fut_for_diffs = []
	i = 0
	while i  < len(maturities_days_temp):

		mean_numbers = 0 

		#if the number is a duplicate, take the mean of all and save it in list of maturities
		if len(np.where(maturities_days_temp == maturities_days_temp[i])[0]) >1:
			mean_index = np.where(maturities_days_temp == maturities_days_temp[i])[0]

			diffs = [fut_for_diffs_temp[j] for j in mean_index]
			meanDiffs = float(sum(diffs))/len(mean_index)



			fut_for_diffs.append(meanDiffs)
			maturities_days.append(maturities_days_temp[i])
		
			i +=len(mean_index)

	
		else: #value is unique, save it in list of maturities
			fut_for_diffs.append(fut_for_diffs_temp[i])
			maturities_days.append(maturities_days_temp[i])
			i += 1


	return zip(*zip(maturities_days,fut_for_diffs))


# ------------------For getting interest rate data-----------------------------

def getInterestRateDates(EONIA, FFE, GCCU):
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
	if GCCU :
		print "calculations using FFE rates"
		pathToData = 'Data/OIS_Data.xlsx'
		sheet = 'FFE_MID'
		#maturitiesInterest = loadFromHDF5('EONIAask.hdf5','times') #maturities, column wise of interest data
		datesInterest = pd.read_excel(pathToData,sheet ,parse_cols = [0]) #dates, row wise of interest rate data
		datesInterest['dates'] = pd.to_datetime(datesInterest['dates'])  # pandas.core.frame.DataFrame of 'numpy.datetime64'> elements
		return datesInterest


	else:
		print "no interest rate choosen"

def getInterestRates():
	return loadFromHDF5('EONIAask.hdf5','ZCMat') #interest rates data, type


# ------------------Getting data-----------------------------

if GOLD:
	#interest rate dates and data
	datesInterest = getInterestRateDates(EONIA = False, FFE = False, GCCU = True)
	ZeroCouponMat = getInterestRates()

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
	datesInterest = getInterestRateDates(EONIA = False, FFE = False, GCCU = True)
	ZeroCouponMat = getInterestRates()


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
	datesInterest = getInterestRateDates(EONIA = True, FFE = False, GCCU = False)
	ZeroCouponMat = getInterestRates()

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
		






