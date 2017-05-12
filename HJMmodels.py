#!/usr/bin/env python
"""
   Stochastic models for interest rate term structures

    v0.1 - Mans Skytt
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from h5pyStorage import loadFromHDF5, storeToHDF5
from forwardCurves import runSurfPlot
"""
#	Helper function for checking timesteps
"""
def checkTimesteps(timesteps):
	if timesteps % 1 != 0:
		print 'Timesteps not even. type(T) = int => T = years, otherwise use T = days/365 '
		return None
	else:
		timestepsInt = int(timesteps)
		return timestepsInt

"""
#	Brownian motion
"""
def genBrownian(T, dt):
	"""
	T : Time until end, year base
	returns:
	dW : array with brownian motion from 0 to T with step size dt
	"""
	timesteps = checkTimesteps(T/dt)
	if timesteps:
		dW = np.random.normal(0,np.sqrt(dt),timesteps)
		return dW
	return None

def genXBrownians(T, dt, X):
	"""
	T : Time until end, year base
	X : Number of brownian motions
	returns:
	dW : array with X brownian motions from 0 to T with step size dt
	"""
	timesteps = checkTimesteps(T/dt)
	dW = genBrownian(T, dt)
	i = 1
	while i < X:
		dWtemp = genBrownian(T, dt)
		dW = np.append(dW, dWtemp)
		i += 1
	dW = np.reshape(dW, (-1,X))
	return dW

"""
#	Two-factor HJM
"""
def twoFactorHJM(initVec, endTime):
	"""
	endTime : Time until stop of simulation, in years
	initVec : Initial forward interest rate term structure
	returns: Time series of forecasted term strucutres
	"""
	sigma1 = 0.005 # Winged volatility constants
	sigma2 = 0.01
	kappa = 0.1
	dt = 1/365 # time horizion time step 
	simdt = 1/365 # simulation timestep
	T = 10
	horizonTimesteps = initVec.shape[0] # Timesteps in horizon
	simTimesteps = checkTimesteps(endTime/simdt) # Timesteps for simulation
	if checkTimesteps:
		# Generate volatility and drift functions
		t = np.linspace(7/365, T, horizonTimesteps) # time horizon
		sigmaf1 = sigma1*np.ones(t.shape) # first volatility function, constant over entire horizon
		sigmaf2 = sigma2*np.exp(-kappa*t) # second volatility function, varying over horizion
		mu = sigma1**2*t + sigma2**2/kappa*np.exp(-kappa*t)*(1-np.exp(-kappa*t)) # drift function
		# Generate brownian motions until end of simulation

		dW1 = genBrownian(endTime, simdt) 
		dW2 = genBrownian(endTime, simdt) 
		# Simulate yield curve progression
		dphi = np.zeros((simTimesteps+1,initVec.shape[0]))# Pre allocate dphi array space
		dphi[0,:] = initVec # Add initial Vector as first
		for i in range(1, simTimesteps):
			dphi[i,:] = mu*dt + sigmaf1*dW1[i] + sigmaf2*dW2[i]
		return np.cumsum(dphi, axis=0) 

def PCHJM(initVec, endTime, PCsMat):
	"""
	endTime : Time until stop of simulation, in years
	initVec : Initial forward interest rate term structure
	PCsMat : Principal components equivalent of the volatility functions
	returns: Time series of forecasted term strucutres
	"""
	dt = 1/365 # time horizion time step 
	simdt = 1/365 # simulation timestep
	T = 10
	numbFactors = PCsMat.shape[1]
	horizonTimesteps = initVec.shape[0] # Timesteps in horizon
	simTimesteps = checkTimesteps(endTime/simdt) # Timesteps for simulation
	if checkTimesteps:
		t = np.linspace(7/365, T, horizonTimesteps) # time horizon
		integratedSigma = np.cumsum(PCsMat, axis=0) # row equal time horizon index, column equal sigma
		mu = np.sum(integratedSigma*PCsMat, axis=1) # 
		dW = genXBrownians(endTime, simdt, numbFactors) 
		
		# Simulate yield curve progression
		dphi = np.zeros((simTimesteps,initVec.shape[0]))# Pre allocate dphi array space
		dphi[0,:] = initVec # Add initial Vector as first
		for i in range(1, simTimesteps):
			sigmaTemp = np.sum(PCsMat*dW[i,:], axis = 1)
			dphi[i,:] = mu*dt + sigmaTemp
	print dphi.shape
	return np.cumsum(dphi, axis = 0)

MATLABForwardMat = loadFromHDF5('EONIAask.hdf5','MATLABForwardMat')
MATLABForwardVec = MATLABForwardMat[0,:]
times = loadFromHDF5('EONIAask.hdf5','times')
MATLABForPCs = loadFromHDF5('EONIAask.hdf5', 'MATLABForPCs')
phi = PCHJM(MATLABForwardVec, 5, MATLABForPCs)
plt.plot(MATLABForPCs)
plt.show()
#phi = twoFactorHJM(forwardVec, 8)
#runSurfPlot(phi, times)












