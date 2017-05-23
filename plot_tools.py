import matplotlib 
import matplotlib.pyplot as plt
matplotlib.interactive(True)



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

def surfPlot():
	pass
	# titleString = "realised difference in futures and forwards"
	# fig_surf = plt.figure()
 #    ax = fig.gca(projection='3d')
 #    y = np.arange(0,forwardMat.shape[0],1)
 #    X, Y = np.meshgrid(times, y)
 #    surf = ax.plot_surface(X, Y, forwardMat, cmap=cm.coolwarm, linewidth=0, antialiased=False)
 #    fig.colorbar(surf, shrink=0.5, aspect=5)
 #    plt.title(titleString)
 #    plt.show()
