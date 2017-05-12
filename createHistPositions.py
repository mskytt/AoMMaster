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

def getMaturity_inYears(col_name,dates):
	return  float((dates.loc[col_name, u'FINAL SETTLE DATE'] -  dates.loc[col_name, u'FUT.1ST PRICE DATE']).days)/252

	



# --------------- start program ---------------

# CHOOSE COMMODITY
gold = True
aluminimum = False
oil = False
power = False
OIS_fed = False
EONIA = False

# -----------------------------------------------
if gold:
	#data
	pathToData =  'Data/GoldFutures.xlsx'
	sheet_specs = 'COMEXGoldFutSpecs'
	index_specs = 0
	datesFutures = pd.read_excel(pathToData,sheet_specs, index_col =index_specs ,parse_cols = [0,2,3])


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
	if os.path.isfile('PowerFutures.hdf5'):
		dfFuturesData = loadFromHDF5('PowerFutures.hdf5', 'PowerFutures')
	else:
		pathToData = 'Data/PowerFutures.xlsx'
		sheets = ['ReutersNordpoolPowerTS_1','ReutersNordpoolPowerTS_2']
		indexColumn = 0
		print "reading sheet  " + str(sheet)

		dfFuturesData = xlExtract.extractData(xlExtract(pathToData,sheet[0],indexColumn), xlsFuturesData.columns,'24-03-2017',  entireTS = True, useLinterpDF = True)
		dfFuturesData_new = dfFuturesData.join(xlExtract.extractData(xlExtract(pathToData,sheet[1],indexColumn), xlsFuturesData.columns,'24-03-2017',  entireTS = True, useLinterpDF = True))
		storeToHDF5('PowerFutures.hdf5','PowerFutures', dfFuturesData)		
		EONIA = True
if EONIA:
	pathToInterestData = 'Data/OIS_data.xlsx'
	interestSheet = 'EONIA_MID'
	interestIndexColumn = i
	maturityPerColumns_year = [1/52, 2/52,3/52,1/12,2/12,3/12,4/12,5/12,6/12,7/12,8/12,9/12,10/12,11/12,1,15/12,18/12,21/12,2,3,4,5,6,7,8,9,10]
	xlsInterestData =xlExtract(pathToInterestData,interestSheet,interestIndexColumn) 
	dfInterestData = xlExtract.extractData(xlsInterestData, xlsInterestData.columns,'2017-04-21',  entireTS = True, useLinterpDF = True)

	
if OIS_fed:

	pathToInterestData = 'Data/OIS_data.xlsx'
	interestSheet = 'FFE_MID'
	interestIndexColumn = 0
	maturityPerColumns_year = [1/52, 1/12,2/12,2/52,3/12,3/52,4/12,5/12,6/12,7/12,8/12,9/12,10/12,11/12,1,2]
	xlsInterestData =xlExtract(pathToInterestData,interestSheet,interestIndexColumn) 
	dfInterestData = xlExtract.extractData(xlsInterestData, xlsInterestData.columns,'2017-04-21',  entireTS = True, useLinterpDF = True)


#remove nans fro


#for interest-futures pairs!
# # ----- FUTURES -----
for column in dfFuturesData.columns:
	oneFutures = dfFuturesData[column].dropna() #remove all NANs from calculations
	TimeToMat = getMaturity_inYears(column,datesFutures)
	dates = oneFutures.index
	interstRates = []
	#find matching interest rate


	
	for date in oneFutures.index:
		dfInterestData.loc[]
		timeToMat -= 1/252


#goal: return the interest rate to invest in
def findInterestRate_perDay(date,timeToMat):
	#choose column
	for maturity in maturityPerColumns_year:
		if 	timeToMat k


	#choose row
	dfInterestData.loc[date,:]



# # ----- BONDS -----
# dfColumnBond = dfBondsData.columns[i]
# dfBondPrices = xlExtract.extractData(dfBondsData, dfColumnBond,'2017-04-21' , entireTS = True, useLinterpDF = True).dropna()


