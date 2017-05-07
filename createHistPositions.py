from xlExtract import xlExtract


class ResultingGain(self):

	def __init__(self,futuresPrices,bondPrices):

			createForwardPosition(futuresPrice[0],futuresPrice[-1])


	def createForwardPosition(strikePrice, maturityPrice):
		self.ForwardGain = maturityPrice - strikePrice

	def createFuturesPosition(futurePrices,bondPrices):
		strike = futuresPrices[0]


		for i in xrange(len(futuresPrices)):
			timeToMat = (futuresPrices.index[-1] - futuresPrices.index[i])/252
			self.futuresPosition += (futuresPrices.values[i] - strike)*bondPrices.values[i]*timeToMat

