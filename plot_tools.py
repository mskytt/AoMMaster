import matplotlib 
import matplotlib.pyplot as plt



def plot_diffs(startDates, endDate, diffs, nameOfFuture):
	print startDates[0:10]
	print startDates[-1]
	print endDate
	titleString = "Realised difference between futures and forwards in futures contract " + nameOfFuture
	plt.title(titleString)
	plt.xlabel("Time of entering contracts")
	plt.ylabel("Difference in USD")
	line1 = plt.plot(startDates,diffs)
	#legend = plt.legend(handles=[line1], loc=1)
	# beautify the x-labels
	plt.gcf().autofmt_xdate()
	plt.show()

def plot_prices(dates, prices, nameOfFuture):
	print dates[0:10]
	print dates[-1]
	titleString = "Futures contract price" + nameOfFuture
	plt.title(titleString)
	plt.xlabel("Time")
	plt.ylabel("Price in USD")
	line1 = plt.plot(dates,prices)
	#legend = plt.legend(handles=[line1], loc=1)
	# beautify the x-labels
	plt.gcf().autofmt_xdate()
	plt.show()

