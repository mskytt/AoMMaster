from xlExtract import xlExtract
import pandas as pd
import numpy as np
from h5pyStorage import storeToHDF5, loadFromHDF5
import os.path
from pdb import set_trace
import datetime

	

def getForwardPosition(strikePrice, maturityPrice):
	return  maturityPrice - strikePrice

def getFuturesPosition(futurePrices,interestRates):
	strike = futuresPrices[0]

	for i in xrange(len(futuresPrices),interestRates):
		timeToMat = getMaturity(col_name,dates)
		futuresPosition += (futuresPrices.values[i] - strike)*interestRates.values[i]*timeToMat
	return futuresPosition

def getMaturity_inDays(col_name,dates):
	return  (dates.loc[col_name, u'FINAL SETTLE DATE'] -  dates.loc[col_name, u'FUT.1ST PRICE DATE']).days

	



# --------------- start program ---------------

# CHOOSE COMMODITY
gold = True
aluminimum = False
oil = False
power = False
EONIA = True


# -----------------------------------------------
if gold:
	#data
	pathToData =  'Data/GoldFutures.xlsx'
	sheet_specs = 'COMEXGoldFutSpecs'
	index_specs = 0
	dfMaturitiesFutures = pd.read_excel(pathToData,sheet_specs, index_col =index_specs ,parse_cols = [0,2,3])


	sheets = ['ReutersCOMEXGoldTS1', 'ReutersCOMEXGoldTS2', 'ReutersCOMEXGoldTS3']
	indexColumn = 0
	dfFuturesData = pd.DataFrame()
	for sheet in sheets[0:1]:
		xlsFuturesData = xlExtract(pathToData,sheet,indexColumn)
		dfFuturesData = dfFuturesData.join(xlExtract.extractData(xlsFuturesData, xlsFuturesData.columns,'2017-04-21',  entireTS = True, useLinterpDF = True).dropna(how = 'all'),how = 'outer')
		print "sheet " + str(sheet) + " extracted"	

	 	
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
	dfFuturesData_new = dfFuturesData.join(xlExtract.extractData(xlExtract(pathToData,sheet[1],indexColumn), xlsFuturesData.columns,'24-03-2017',  entireTS = True, useLinterpDF = True))
	storeToHDF5('PowerFutures.hdf5','PowerFutures', dfFuturesData)		
	EONIA = True

if EONIA:
	pathToData = 'Data/OIS_Data.xlsx'
	sheet = 'EONIA_MID'
	ForwarRatesdMat = loadFromHDF5('EONIAask.hdf5','MATLABForwardMat') #interest rates data, type
	maturitiesInterest = loadFromHDF5('EONIAask.hdf5','times') #maturities, column wise of interest data
	datesInterest = pd.read_excel(pathToData,sheet ,parse_cols = [0]) #dates, row wise of interest rate data
	datesInterest['dates'] = pd.to_datetime(datesInterest['dates']) 




# ----- FUTURES -----
#for column in dfFuturesData.columns:
column = dfFuturesData.columns[0]
oneFutures = dfFuturesData[column].dropna() #remove all NANs from calculations
timeToMat = getMaturity_inDays(column,dfMaturitiesFutures)
datesFutures = oneFutures.index

# # print "------- dates interest --------"
# # print datesInterest

# # print "------- datesInterest.index(date) --------"
# # date = datesFutures[0]

# date = datesInterest.iloc[3] 
# print type(date.values)
# print date.values
# # print "-------   date = datesInterest.iloc[3] --------"

# #print datesInterest.loc['2016-08-01']
# # print date.values
# #print datesInterest[datesInterest.values == date.values].index.tolist()


# 	#find matching interest rate

# #goal: return the interest rate to invest in

interestRates = []
ForwarRatesdMat_index = []
for date in datesFutures[0:1]:
	print type(date)
	print date

	#print date

	print ForwarRatesdMat_index.append(datesInterest[datesInterest.values == date].index.tolist())
	#interestRates = ForwarRatesdMat(datesFutures.index)




	





