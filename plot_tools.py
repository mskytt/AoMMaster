import matplotlib 
import matplotlib.pyplot as plt
matplotlib.interactive(True)
import numpy as np
from pdb import set_trace

# -------------------------------------------------------------
#
#        Main
#
# ---------------------------------------------------------------

def _onePlotPerFuture(dates, diffs, values, prices, forward_value, interestRates, nameOfFuture):


	#plot_diffs(dates, diffs, nameOfFuture)
	save_diff(dates, values, forward_value, interestRates, prices, nameOfFuture)
	#plot_prices(dates, prices, nameOfFuture)
	#plot_summary_per_future(dates, values, forward_value, interestRates, prices, nameOfFuture)


def _summaryPlot(titleString,saveString, startDates, diffs, maturities_days,interestRates_at_startDates):

	if saveString[-12:] == "all_mats.png": #ALL DATA
		print "Plotting summary plot for all data"
		saveString_mean = saveString[:-12] + "mean" + saveString[-12:]
		#take mean of diffs over maturities
		maturities_days_mean, diffs_mean = takeMeanOfAllEqualMaturitiesDiffs(maturities_days, diffs)
		plot_mean_diffs_summary_mat(titleString,saveString, diffs_mean,maturities_days_mean,  maturities_days)


	else:
		startDates_mean, diffs_mean = takeMeanofSameDateDiffs(startDates,diffs)
		print "Plotting summary plot for all maturity group"
		plot_diffs_per_mat_group_dates(titleString, saveString, startDates,diffs,interestRates_at_startDates)



# -------------------------------------------------------------
#
#        Modify data for plotting
#
# ---------------------------------------------------------------

		
def takeMeanOfAllEqualMaturitiesDiffs(maturities_days, fut_for_diffs):


	#convert arrays into np arrays in order to use specific np function
	maturities_days_temp = np.asarray(maturities_days)
	fut_for_diffs_temp = np.asarray(fut_for_diffs)


	maturities_days = []
	fut_for_diffs = []
	fut_for_diffs_std = []

	i = 0
	while i  < len(maturities_days_temp):

		mean_numbers = 0 

		#if the number is a duplicate, take the mean of all and save it in list of maturities
		if len(np.where(maturities_days_temp == maturities_days_temp[i])[0]) >1:
			mean_index = np.where(maturities_days_temp == maturities_days_temp[i])[0]

			diffs = [fut_for_diffs_temp[j] for j in mean_index]
			meanDiffs = np.mean(diffs)
			stdDiffs = np.std(diffs)



			fut_for_diffs.append(meanDiffs)
			fut_for_diffs_std.append(volDiffs)
			maturities_days.append(maturities_days_temp[i])
		
			i +=len(mean_index)

	
		else: #value is unique, save it in list of maturities
			fut_for_diffs.append(fut_for_diffs_temp[i])
			maturities_days.append(maturities_days_temp[i])

			i += 1


	return zip(*zip(maturities_days,fut_for_diffs))


def takeMeanofSameDateDiffs(startdates, fut_for_diffs):
	print "inside takeMeanofSameDateDiffs"

	#convert arrays into np arrays in order to use specific np function
	startdates_temp = np.asarray(startdates)
	fut_for_diffs_temp = np.asarray(fut_for_diffs)


	startdates = []
	fut_for_diffs = []
	i = 0
	while i  < len(startdates_temp):

		mean_numbers = 0 
		timeDelta = abs(startdates_temp - startdates_temp[i])
		timeDelta = timeDelta.astype('timedelta64[D]').astype(int)

		#if the date is very near another date, take the mean of that value 
		if len(timeDelta[timeDelta < 63]) > 1:
			print len(timeDelta[timeDelta < 63])

			mean_index = np.where(timeDelta[timeDelta < 63])
			print mean_index


			diffs = [fut_for_diffs_temp[j] for j in mean_index]
			print diffs
			meanDiffs = np.mean(diffs)
			print meanDiffs




			fut_for_diffs.append(meanDiffs)
			startdates.append(startdates_temp[i])
		
			i +=len(mean_index)


		else: #value is unique, save it in list of maturities
			fut_for_diffs.append(fut_for_diffs_temp[i])
			startdates.append(startdates_temp[i])
			i += 1



	return zip(*zip(startdates,fut_for_diffs))





#--------------------------------------------------------------


def save_diff(dates, values, forward_value, interestRates, prices, nameOfFuture):
	startDate = dates[0]
	endDate = dates[-1]
	name = nameOfFuture

	#diff
	fut_forDiffs = [float(fut_i) - float(for_i) for fut_i, for_i in zip(futureValues, forwardValues)]
	final_diff = futureValues[-1] - forwardValues[-1]

	#interest rates
	interestRate_mean = np.mean(interestRates)
	interestRate_mean = np.mean(interestRates)
	interestRates_std = np.std(interestRates)

	futuresPrice_mean = np.mean(prices)
	futuresPrice_std = np.std(prices)


	textString = str(nameOfFuture) + " :Diff: " +  str(final_diff) + " Mean Interest rate: " + str(interestRate_mean) + "Vol Interest rate: " + str(interestRate_std) + "Mean futures price: " + str(futuresPrice_mean) + "Vol futures price: " + str(futuresPrice_std) 
				

 	with open('Data/Details_per_futurePos.txt', 'a') as file:
		file.write(textString)



# ----------------- onePlotPerFuture ------------------------



# ----------------- summaryPlots ------------------------
#def plot_diffs_volatilePeriod()


def plot_diffs_summary_mat(titleString,saveString, diffs,maturities_days):

	fig5 = plt.figure()
	ax_left = plt.gca()
	ax_right = ax_left.twinx()



	ax_left.set_ylim(min(diffs), max(diffs))
	ax_left.plot(maturities_days,diffs,'ro')

	#plt.xticks(np.arange(min(maturities_days), max(maturities_days)+1, 1.0))
	#plt.xticks(maturities_days)
	#ax_right.set_xticks(maturities_days, minor=True)
	
	plt.title(titleString)
	plt.xlabel("Number of contracts")
	plt.ylabel("Difference in % from forwardprice")
	plt.savefig(saveString)



def plot_mean_diffs_summary_mat(titleString,saveString, diffs_mean,maturities_days_mean,  maturities_days):

	fig5 = plt.figure()
	ax_left = plt.gca()
	ax_right = ax_left.twinx()

	bins = [maturities_days_mean]
	ax_right.hist(maturities_days)

	ax_left.set_ylim(min(diffs), max(diffs))
	ax_left.plot(maturities_days_mean,diffs_mean,'ro')



	#plt.xticks(np.arange(min(maturities_days), max(maturities_days)+1, 1.0))
	#plt.xticks(maturities_days)
	#ax_right.set_xticks(maturities_days, minor=True)
	
	plt.title(titleString)
	ax_left.set_xlabel("Maturity in days")
	ax_left.set_ylabel("Mean difference in % from forwardprice")
	ax_right.set_ylabel("Number of contracts")
	plt.savefig(saveString)


def plot_diffs_per_mat_group_dates(titleString, saveString, startDates,diffs, interestRates_at_startDates):
	print "inside plot_diffs_per_mat_group_dates"
	fig6 = plt.figure()
	ax_left = plt.gca()
	ax_right = ax_left.twinx()
	
####

	ax_left.plot(startDates,diffs, 'bo', label = 'short forward short + long future')
	ax_right.plot(startDates, interestRates_at_startDates, 'r^', label = 'USSG rate with equal maturity')


	#x axis
	ax_left.set_xticks(startDates, minor=True)
	ax_left.set_xlabel("StartDates")
	ax_left.set_ylabel("Differences in % of total price change")
	ax_right.set_ylabel("FFE Rate")
	ax_left.grid(True)

	ax_right.legend(loc='upper right', fontsize =  'small')
	ax_left.legend(loc='upper right', fontsize =  'small')



	# beautify the x-labels
	plt.gcf().autofmt_xdate() 	


	#title and save
	plt.title(titleString)	
	plt.savefig(saveString)









def plot_diffs_summary_dates(startDates, diffs):
	fig1 = plt.figure()
	plt.plot(startDates,diffs)
	titleString = "Realised moving difference between futures and forwards (futures long - forward short)"
	saveString = "Plots/diffs_sameStart.png"
	plt.title(titleString)
	plt.xlabel("Time of entering contracts")
	plt.ylabel("Difference in USD")
	plt.gcf().autofmt_xdate() 	# beautify the x-labels
	plt.savefig(saveString)


#fut_for_diffs,maturities_days, startDates
def surfPlot(mat,dates):
	"""
	Plot a surface of the forward curves.
	the number of days with valid data, for 
	EONIA: 3037, from 2005-08-11 and forward
	FFE <= 1Y mat: 3158, from 2004-12-30 and forward
	FFE <= 2Y mat: 1328, from 2012-01-13 and forward
	"""
	titleString = "realised difference in futures and forwards"


	ax = fig.gca(projection='3d')
	y = np.arange(0,forwardMat.shape[0],1)
	X, Y = np.meshgrid(times, y)
	surf = ax.plot_surface(X, Y, forwardMat, cmap=cm.coolwarm, linewidth=0, antialiased=False)
	fig.colorbar(surf, shrink=0.5, aspect=5)


	plt.ylabel("start dates of contracts")
	plt.xlabel("MAturities of contracts in years")
	plt.zlabel("Difference in USD")
	plt.title(titleString)
	plt.show()


	

def plot_diffs(startDates, diffs, nameOfFuture):
	fig1 = plt.figure()
	plt.plot(startDates,diffs, label = nameOfFuture)
	titleString = "Realised moving difference between futures and forwards (futures long - forward long) \n depending when contract was entered "
	saveString = "Plots/diffs_" + str(nameOfFuture) + ".png"
	plt.title(titleString)
	plt.xlabel("Time of entering contracts")
	plt.ylabel("Difference in USD")
	plt.legend(loc='upper left')
	plt.gcf().autofmt_xdate() 	# beautify the x-labels
	plt.savefig(saveString)

def plot_prices(dates, prices, nameOfFuture):
	fig2 = plt.figure()
	plt.plot(dates,prices, label = nameOfFuture)
	titleString = "Futures contract price"
	saveString = "Plots/price_" + str(nameOfFuture) + ".png"
	plt.title(titleString)
	plt.xlabel("Time")
	plt.ylabel("Price in USD")
	plt.legend(loc='upper left')
	plt.gcf().autofmt_xdate() 	# beautify the x-labels
	plt.savefig(saveString)







def plot_summary_per_future(dates, futureValues, forwardValues, interestRates, prices, nameOfFuture):
	#define fig and axises
	fig3 = plt.figure()
	ax_left = plt.gca()
	ax_right = ax_left.twinx()
	#ax_right.set_ylim([min(interestRates),max(interestRates)])



	titleString =  str(nameOfFuture)
	plt.title(titleString)

	#plot with legends
	ax_left.plot(dates,futureValues, label = 'futures position')
	ax_left.plot(dates, forwardValues,  label = 'forward position')
	#ax_left.plot(dates, prices,'r', label = "Futures price")
		
	ax_right.plot(dates,interestRates, 'g', label = 'interest rate')


	#Axises
	ax_left.set_xlabel("Time")
	ax_left.set_ylabel("Value in USD")
	ax_right.set_ylabel("Zero coupon rate")

	plot.legend(loc='upper left')
	#ax_right.legend(loc='upper right')
	# beautify the x-labels
	plt.gcf().autofmt_xdate() 	


	saveString = "Plots/summary_" + str(nameOfFuture) + ".png"
	plt.savefig(saveString)




   