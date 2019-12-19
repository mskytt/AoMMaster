#!/usr/bin/env python
"""
    Stochastic models for interest rate term structures
 	Run script to simulate. Option to replace input with own input. See bottom of file
    v0.1 - Mans Skytt (m@skytt.eu)
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
		return int(round(timesteps))
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
	sigma1 = 0.005 # volatility constants, replace with estimate made w suitable approximation (ML or kalman)
	sigma2 = 0.01
	kappa = 0.1
	dt = 1/365 # time horizion time step 
	simdt = 7/365 # simulation timestep
	T = 10
	horizonTimesteps = initVec.shape[0] # Timesteps in horizon
	simTimesteps = checkTimesteps(endTime/simdt) # Timesteps for simulation
	# Generate volatility and drift functions
	t = np.linspace(7/365, T, horizonTimesteps) # time horizon
	sigmaf1 = sigma1*np.ones(t.shape) # first volatility function, constant over entire horizon
	sigmaf2 = sigma2*np.exp(-kappa*t) # second volatility function, varying over horizion
	mu = sigma1**2*t + sigma2**2/kappa*np.exp(-kappa*t)*(1-np.exp(-kappa*t)) # drift function
	# Generate brownian motions until end of simulation

	dW1 = genBrownian(endTime, simdt) 
	dW2 = genBrownian(endTime, simdt) 
	# Simulate yield curve progression
	dphi = np.zeros((simTimesteps,initVec.shape[0]))# Pre allocate dphi array space
	dphi[0,:] = initVec # Add initial Vector as first
	for i in range(1, simTimesteps):
		dphi[i,:] = mu*dt + sigmaf1*dW1[i] + sigmaf2*dW2[i]
	return np.cumsum(dphi, axis=0) 

"""
#	PC-based HJM
"""
def PCHJM(initVec, endTime, PCsMat):
	"""
	endTime : Time until stop of simulation, in years
	initVec : Initial forward interest rate term structure
	PCsMat : Principal components equivalent of the volatility functions. Could also be volatility functions for every time horizon
	returns: Time series of forecasted term strucutres
	"""
	dt = 1/365 # time horizion time step 
	simdt = 7/365 # simulation timestep
	T = 10 # Time horizon max
	numbFactors = PCsMat.shape[1]
	horizonTimesteps = initVec.shape[0] # Timesteps in horizon
	simTimesteps = checkTimesteps(endTime/simdt) # Timesteps for simulation

	integratedSigma = np.cumsum(PCsMat, axis=0) # row equal time horizon index, column equal different sigmas
	mu = np.sum(integratedSigma*PCsMat, axis=1) # 
	dW = genXBrownians(endTime, simdt, numbFactors) 
	# Simulate yield curve progression
	dphi = np.zeros((simTimesteps,initVec.shape[0]))# Pre allocate dphi array space
	dphi[0,:] = initVec # Add initial Vector as first
	for i in range(1, simTimesteps):
		sigmaTemp = np.sum(np.multiply(PCsMat,dW[i,:]), axis = 1)*np.sqrt(252)
		dphi[i,:] = mu*dt + sigmaTemp
	return np.cumsum(dphi, axis = 0)

def HoLee(initVec, endTime, vol):
	"""
	endTime : Time until stop of simulation, in years
	initVec : Initial forward interest rate term structure
	vol : volatility, yearly basis
	returns: Time series of forecasted term strucutres
	"""
	dt = 1/365 # time horizion time step 
	simdt = 7/365 # simulation timestep
	T = 10
	horizonTimesteps = initVec.shape[0] # Timesteps in horizon
	simTimesteps = checkTimesteps(endTime/simdt) # Timesteps for simulation
	# Generate volatility and drift functions
	t = np.linspace(7/365, T, horizonTimesteps) # time horizon
	volf = vol*np.ones(t.shape) # volatility, constant over entire horizon
	mu = vol**2*t # drift function
	# Generate brownian motions until end of simulation
	dW = genBrownian(endTime, simdt) 
	
	# Simulate yield curve progression
	dphi = np.zeros((simTimesteps,initVec.shape[0]))# Pre allocate dphi array space
	dphi[0,:] = initVec # Add initial Vector as first
	for i in range(1, simTimesteps):
		dphi[i,:] = mu*dt + volf*dW[i]
	return np.cumsum(dphi, axis=0) 



# Load from storage file
storageFile = 'EONIAmid.hdf5' # Name of file where data is currently stored
MATLABForwardMat = loadFromHDF5(storageFile,'MATLABForwardMat')
MATLABForwardVec = MATLABForwardMat[0,:]
times = loadFromHDF5(storageFile,'times')
MATLABForPCs = loadFromHDF5(storageFile, 'MATLABForPCs')

# Run simulations, comment out the ones not used
phi = PCHJM(MATLABForwardVec, 5, MATLABForPCs)
# phi = twoFactorHJM(MATLABForwardVec, 8)
#phi = HoLee(MATLABForwardVec, 5, 0.1)

runSurfPlot(phi[:,:times.shape[0]], times)
plt.show()












