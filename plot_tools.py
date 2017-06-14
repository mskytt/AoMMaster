import matplotlib 
import matplotlib.pyplot as plt
matplotlib.interactive(True)
import numpy as np


def _onePlotPerFuture(dates, diffs, values, prices, forward_value, interestRates, nameOfFuture):


	#plot_diffs(dates, diffs, nameOfFuture)
	save_diff(dates, values, forward_value, nameOfFuture)
	#plot_prices(dates, prices, nameOfFuture)
	plot_summary_per_future(dates, values, forward_value, interestRates, prices, nameOfFuture)




def _summaryPlot(titleString,saveString, startDates, diffs, maturities_days,interestRates_at_startDates):
	
	print saveString[-12:]


	if saveString[-12:] == "all_mats.png": #ALL DATA
		print "Plotting summary plot for all data"

		plot_diffs_summary_mat(titleString,saveString, diffs,maturities_days)

	else:
		print "Plotting summary plot for all maturity group"
		plot_diffs_per_mat_group_dates(titleString, saveString, startDates,diffs,interestRates_at_startDates)







#--------------------------------------------------------------


def save_diff(dates, futureValues, forwardValues, nameOfFuture):
	startDate = dates[0]
	endDate = dates[-1]
	name = nameOfFuture
	fut_forDiffs = [float(fut_i) - float(for_i) for fut_i, for_i in zip(futureValues, forwardValues)]

	final_diff = futureValues[-1] - forwardValues[-1]


	#print futureValues
	print "---------------------Final Diff--------------------------"

	print " \n" + str(name) + "\n"
	print  str(futureValues[-1]) + " - " + str(forwardValues[-1])  + " = " + str(final_diff)
	print "\n"

	print "----------------------------------------------------------"


# ----------------- onePlotPerFuture ------------------------



# ----------------- summaryPlots ------------------------
#def plot_diffs_volatilePeriod()


def plot_diffs_summary_mat(titleString,saveString, diffs,maturities_days):

	fig5 = plt.figure()
	ax_left = plt.gca()
	ax_right = ax_left.twinx()
	ax_right.hist(maturities_days)
	ax_left.set_ylim(min(diffs), max(diffs))
	ax_left.plot(maturities_days,diffs,'ro')

	#plt.xticks(np.arange(min(maturities_days), max(maturities_days)+1, 1.0))
	#plt.xticks(maturities_days)
	#ax_right.set_xticks(maturities_days, minor=True)
	
	plt.title(titleString)
	plt.xlabel("Number of contracts")
	plt.ylabel("Difference in % from forwardprice")
	plt.savefig(saveString)



def plot_diffs_per_mat_group_dates(titleString, saveString, startDates,diffs, interestRates_at_startDates):
	print "inside plot_diffs_per_mat_group_dates"
	fig6 = plt.figure()
	ax_left = plt.gca()
	ax_right = ax_left.twinx()
	
####

	ax_left.plot(startDates,diffs, 'bo', label = 'short forward short + long future')
	ax_right.plot(startDates, interestRates_at_startDates, 'r^', label = 'USSG rate with equal maturity')
	plt()

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




   