import matplotlib 
import matplotlib.pyplot as plt
import scipy.interpolate as interp
from mpl_toolkits.mplot3d import Axes3D
matplotlib.interactive(True)
import numpy as np


def plot_diffs(startDates, endDate, diffs, nameOfFuture):
	fig1 = plt.figure()
	plt.plot(startDates,diffs, label = nameOfFuture)
	titleString = "Realised moving difference between futures and forwards (futures long - forward long)"
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


def plot_value(dates, prices, nameOfFuture):
	fig3 = plt.figure()
	plt.plot(dates,prices, label = nameOfFuture)
	titleString = "Futures contract value" 
	saveString = "Plots/value_" + str(nameOfFuture) + ".png"
	plt.title(titleString)
	plt.xlabel("Time")
	plt.ylabel("Value in USD")
	plt.legend(loc='upper left')
	plt.gcf().autofmt_xdate() 	# beautify the x-labels
	plt.savefig(saveString)

def plot_diffs_in_same(startDates, endDate, diffs, nameOfFuture):
	plt.plot(startDates,diffs, label = nameOfFuture)
	titleString = "Realised moving difference between futures and forwards (futures long - forward long)"
	saveString = "Plots/diffs.png"
	plt.title(titleString)
	plt.xlabel("Time of entering contracts")
	plt.ylabel("Difference in USD")
	plt.legend(loc='upper left')
	plt.gcf().autofmt_xdate() 	# beautify the x-labels
	plt.hold(True)
	plt.savefig(saveString)


def plot_diffs_mat(diffs,maturities_days):
	fig5 = plt.figure()
	plt.plot(maturities_days,diffs)
	titleString = "Realised difference between futures and forwards (futures long - forward_short) if entering both contracts at startdates"
	saveString = "Plots/diffs_mat.png"
	plt.title(titleString)
	plt.xlabel("Maturity in days")
	plt.ylabel("Difference in USD")
	plt.savefig(saveString)

def plot_diffs_sameStart(startDates, diffs):
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

   