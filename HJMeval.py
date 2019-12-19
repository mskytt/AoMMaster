#!/usr/bin/env python
"""
   Evaluation of stochastic models for interest rate term structures

    v0.1 - Mans Skytt
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from h5pyStorage import loadFromHDF5, storeToHDF5
from forwardCurves import runGenerateData, runGenZCPCs, runGenForPCs, runGenMatlab, run, runSurfPlot, genEigs, genPCs
from HJMmodels import PCHJM, twoFactorHJM, HoLee


def runPCAHJMSim(initVec, endTime, PCsMat, actualForward, sims):
#	If the data is too much, only save 1 each week
	times = loadFromHDF5(storageFile,'times') # Remove later
	dataMat = np.zeros((sims*7,initVec.shape[0])) # Row 0 - 1 week; Row 1 - 1 month; Row 2 - 6 months; Row 3 - 1 year; Row 4 - 2 years; Row 5 - 5 years; Row 6 - 10 years
	
	print 'Simulating..'
	print 'Simulation number: 1'
	phi = PCHJM(initVec, endTime, PCsMat) # to get shape
	dataMat = np.zeros((sims*7,phi.shape[0])) # Row 0 - 1 week; Row 1 - 1 month; Row 2 - 6 months; Row 3 - 1 year; Row 4 - 2 years; Row 5 - 5 years; Row 6 - 10 years
	dataMat[:7,:] = phi[:,[6, 30, 180, 365, 730, 1825, 3650]].T
	
	for simNumb in range(1,sims): # Make remaining simulations
		print 'Simulation number: ', simNumb+1
		phi = PCHJM(initVec, endTime, PCsMat) # Row equal to days in future from start date, column equal time in horizon
		dataMat[simNumb*7:(simNumb+1)*7,:] = phi[:,[6, 30, 180, 365, 730, 1825, 3650]].T

		# runSurfPlot(phi[:,:times.shape[0]], times) # Remove later
		# plt.title('Simulated forward curves from HJM PCA.')
		# plt.show()
		# runSurfPlot(actualForward[0:endTime*365:7,:times.shape[0]], times)
		# plt.title('Actual forward forwardCurves.')
		# plt.show()
	return dataMat

def runTwoFactorHJMSim(initVec, endTime, actualForward, sims):
#	If the data is too much, only save 1 each week
	times = loadFromHDF5(storageFile,'times') # Remove later
	dataMat = np.zeros((sims*7,initVec.shape[0])) # Row 0 - 1 week; Row 1 - 1 month; Row 2 - 6 months; Row 3 - 1 year; Row 4 - 2 years; Row 5 - 5 years; Row 6 - 10 years
	
	print 'Simulating..'
	print 'Simulation number: 1'
	phi = twoFactorHJM(initVec, endTime) # to get shape
	dataMat = np.zeros((sims*7,phi.shape[0])) # Row 0 - 1 week; Row 1 - 1 month; Row 2 - 6 months; Row 3 - 1 year; Row 4 - 2 years; Row 5 - 5 years; Row 6 - 10 years
	dataMat[:7,:] = phi[:,[6, 30, 180, 365, 730, 1825, 3650]].T
	
	for simNumb in range(1,sims): # Make remaining simulations
		print 'Simulation number: ', simNumb+1
		phi = twoFactorHJM(initVec, endTime) # Row equal to days in future from start date, column equal time in horizon
		dataMat[simNumb*7:(simNumb+1)*7,:] = phi[:,[6, 30, 180, 365, 730, 1825, 3650]].T

		# runSurfPlot(phi[:,:times.shape[0]], times) # Remove later
		# runSurfPlot(actualForward[0:endTime*365,:times.shape[0]], times)
		# plt.show()
	return dataMat

def runHoLeeSim(initVec, endTime, actualForward, vol, sims):
#	If the data is too much, only save 1 each week
	times = loadFromHDF5(storageFile,'times') # Remove later
	dataMat = np.zeros((sims*7,initVec.shape[0])) # Row 0 - 1 week; Row 1 - 1 month; Row 2 - 6 months; Row 3 - 1 year; Row 4 - 2 years; Row 5 - 5 years; Row 6 - 10 years
	
	print 'Simulating..'
	print 'Simulation number: 1'
	phi = HoLee(initVec, endTime, vol) # to get shape
	dataMat = np.zeros((sims*7,phi.shape[0])) # Row 0 - 1 week; Row 1 - 1 month; Row 2 - 6 months; Row 3 - 1 year; Row 4 - 2 years; Row 5 - 5 years; Row 6 - 10 years
	dataMat[:7,:] = phi[:,[6, 30, 180, 365, 730, 1825, 3650]].T
	
	for simNumb in range(1,sims): # Make remaining simulations
		print 'Simulation number: ', simNumb+1
		phi = HoLee(initVec, endTime, vol) # Row equal to days in future from start date, column equal time in horizon
		dataMat[simNumb*7:(simNumb+1)*7,:] = phi[:,[6, 30, 180, 365, 730, 1825, 3650]].T

		# runSurfPlot(phi[:,:times.shape[0]], times) # Remove later
		# runSurfPlot(actualForward[0:endTime*365,:times.shape[0]], times)
		# plt.show()
	return dataMat


def RMSE(simMat, actualMat):
	"""
	#	Compute RMSE of for each sample, i.e. 3M rate
	#	Return: Matrix with rmse of each maturity for each simulation time horizon
	"""
	simulations = simMat.shape[0]/actualMat.shape[0]
	print 'simulations', simulations
	simDiffMat = np.diff(simMat) # Compute changes along each row
	print 'sim mat diff: \n', simDiffMat.shape
	actualMat = np.diff(actualMat)[:,:simDiffMat.shape[1]]
	print 'actual mat diff: \n',  actualMat.shape
	errorMat = np.zeros(simDiffMat.shape) # Allocate matrix for erorrs
	RMSEMat = np.zeros(actualMat.shape)
	for row in range(actualMat.shape[0]):
		errorMat[row::6,:] = simDiffMat[row::6,:]  - actualMat[row,:] 
		RMSEMat[row,:] = np.mean(np.square(simDiffMat[row::6,:]  - actualMat[row,:]), axis=0) # Mean of rows from each sample
	RMSEMat = np.sqrt(RMSEMat) 

	return RMSEMat

sheetName = 'EONIA_MID' # Sheet name
storageFile = 'EONIAmid.hdf5' # Name of file where data is to be/ is currently stored
MATLABstorageFile = 'MatlabEONIA05midForward100.hdf5' 

# Bools to generate new
genMatlab = True
genMatlabEigs = True

endTime = 2 # Years of simulation
startTime = 1500 # Start place of simulation
MATLABForwardMat = loadFromHDF5(MATLABstorageFile,'MATLABFordataMat')
runGenMatlab(genMatlab, genMatlabEigs, MATLABForwardMat[:,startTime-1000:startTime], sheetName, storageFile) 
MATLABForEigVecs = loadFromHDF5(storageFile,'MATLABForEigVecs')
MATLABForPCs = loadFromHDF5(storageFile, 'MATLABForPCs')
times = loadFromHDF5(storageFile,'times')
MATLABForEigPerc = loadFromHDF5(storageFile, 'MATLABForEigPerc')
print MATLABForPCs.shape
i = 1
for column in range(MATLABForPCs.shape[1]):
	plt.plot(MATLABForPCs[:,column],label='Component: '+str(i)+', perc:'+str(round(MATLABForEigPerc[column]*100,2)) + '%')
	i += 1
plt.title('Principal components generated from 1000 days data')
plt.legend(loc='best')
plt.show()
# plt.savefig('PCs.png', bbox_inches='tight')
# plt.clf()

i = 1
for column in range(MATLABForPCs.shape[1]):
	plt.plot(MATLABForEigVecs[:,column],label='Component: '+str(i))
	i += 1
plt.title('Eigen-vectors generated from 1000 days data')
plt.legend(loc='best')
plt.show()
# plt.savefig('eigs.png', bbox_inches='tight')

# horizonVec = [6, 30, 180 ,365, 730, 1825, 3650]
# numbOfSims = 1000
# compForwardMat = MATLABForwardMat.T
# actualForward = compForwardMat[startTime:startTime+endTime*365,horizonVec].T
# simTSMatPCA = runPCAHJMSim(MATLABForwardMat[:,startTime], endTime, MATLABForPCs, compForwardMat[startTime:,:], numbOfSims)
# simTSMat2HJM = runTwoFactorHJMSim(MATLABForwardMat[:,startTime], endTime, compForwardMat[startTime:,:], numbOfSims)
# simTSMatHoLee = runHoLeeSim(MATLABForwardMat[:,startTime], endTime, compForwardMat[startTime:,:], 0.01, numbOfSims)
# RMSEMatPCA = RMSE(simTSMatPCA, actualForward[:,::7])
# RMSEMat2HJM = RMSE(simTSMat2HJM, actualForward[:,::7])
# RMSEMatHoLee = RMSE(simTSMatHoLee, actualForward[:,::7])

# percPCAHoLee = RMSEMatPCA/RMSEMatHoLee
# perc2HJMHoLee = RMSEMat2HJM/RMSEMatHoLee
# percPCA2HJM = RMSEMatPCA/RMSEMat2HJM

#simTime = 150 # weeks

# plt.plot(horizonVec, RMSEMatPCA[:,simTime], 'bo', label='HJM PCA model')
# plt.plot(horizonVec, RMSEMat2HJM[:,simTime], 'r*', label='HJM Two factor model')
# plt.plot(horizonVec, RMSEMatHoLee[:,simTime], 'gv', label='Ho-Lee model')
# plt.title('Absolute RMSE for ' + str(simTime*7) + ' days forecast from ' + str(numbOfSims) + ' simulations')
# plt.legend(loc='best')
# plt.xlabel('Maturity (Days)')

# plt.ylabel('RMSE')
# # plt.show()
# simTimes = [1, 20, 50, endTime*50-endTime] # weeks
# for simTime in simTimes:
# 	plt.plot(horizonVec, percPCAHoLee[:,simTime], 'bo', label='HJM PCA / Ho-Lee')
# 	plt.plot(horizonVec, perc2HJMHoLee[:,simTime], 'r*', label='HJM Two factor / Ho-Lee')
# 	plt.plot(horizonVec, percPCA2HJM[:,simTime], 'go', label='HJM PCA / HJM Two factor')
# 	plt.title('Relative RMSE for ' + str(simTime*7) + ' days forecast from ' + str(numbOfSims) + ' simulations.' )
# 	plt.legend(loc='best')
# 	plt.xlabel('Maturity (Days)')
# 	plt.ylabel('Relative RMSE')
# 	plt.savefig(str(startTime)+'_'+str(simTime)+'RelativeRMSE.png', bbox_inches='tight')#show()
# 	plt.clf()

# simTimes = [1, 20, 50, endTime*50-endTime]
# for simTime in simTimes:
# 	plt.plot(horizonVec, RMSEMatPCA[:,simTime], label='T = '+str(simTime))

# plt.title('Absolute RMSE for the HJM PCA model. ' + str(numbOfSims) + ' simulations')
# plt.legend(loc='best')
# plt.xlabel('Maturity (Days)')
# plt.ylabel('RMSE')
# plt.savefig(str(startTime)+'PCAAbsoluteRMSE.png', bbox_inches='tight')

# # plt.plot(simTSMat[3::7,:].T, color='0.2', alpha=0.1)
# plt.plot(actualForward[3,::7].T, color='r')
# plt.show()

# plt.plot(simTSMat2[3::7,:].T, color='0.2', alpha=0.1)
# plt.plot(actualForward[3,::7].T, color='r')
# plt.show()
# plt.plot(MATLABForPCs)
# plt.show()
# phi = twoFactorHJM(MATLABForwardVec, 8)
# runSurfPlot(phi[:,:times.shape[0]], times)

