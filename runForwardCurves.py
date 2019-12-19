"""
	Template on how to run the forward curve computations
"""

from forwardCurves import runGenerateData, runGenZCPCs, runGenForPCs, runGenMatlab, run, runSurfPlot
from h5pyStorage import loadFromHDF5
import numpy as np
"""
    Set bools to choose where to get data from
"""
readExcel = True # Read from excel
genForward = True # Generate forward rates matrix
genZC = True # Generate zero coupon rates matrix
genForEigs = True # Generate forward eigenvalues
genZCEigs = True # Generate zero coupon eigenvalues
genMatlab = True
genMatlabEigs = True

sheetName = 'EONIA_MID' # Sheet name
storageFile = 'EONIAmid.hdf5' # Name of file where data is to be/ is currently stored
MATLABstorageFile = 'MatlabEONIA05midForward100.hdf5'  # Name of Matlab file to use

"""
	Different matlab generated hdf5 files: 

	'MatlabEONIAmidForward100.hdf5' -
	'MatlabEONIA05midForward100.hdf5' -
	'MatlabEONIA05midForward1000.hdf5' -
	'MatlabFFE2YmidForward100.hdf5' - 
	'MatlabFFE2Y05midForward100.hdf5' -
	'MatlabFFE2Y025midForward100.hdf5' -
	'MatlabFFE2Y025midForward1000.hdf5' - 
"""
"""
    Run functions
"""
runGenerateData(readExcel, genForward, genZC, sheetName, storageFile) # Generates data from excel sheet using 'only' cubic splines
# ZCMatUSGG = loadFromHDF5(storageFile,'ZCMat')
# ZCMatFFE = loadFromHDF5('FFEmid.hdf5','ZCMat')
# ZCMat = loadFromHDF5(storageFile,'ZCMat')
# print ZCMatFFE.shape, ZCMatUSGG.shape
# combined = np.column_stack((ZCMatFFE[:100,:23],ZCMatUSGG[:100,:100]))
# runSurfPlot(combined, range(combined.shape[1])) 
# ZCMatDiff = loadFromHDF5(storageFile,'ZCMatDiff')
# runGenZCPCs(genZCEigs, ZCMatDiff, storageFile)
# forMatDiff = loadFromHDF5(storageFile,'forMatDiff')
# runGenForPCs(genForEigs, forMatDiff, storageFile)
MATLABForwardMat = loadFromHDF5(MATLABstorageFile,'MATLABFordataMat')
runGenMatlab(genMatlab, genMatlabEigs, MATLABForwardMat[:,0:3000], sheetName, storageFile) # Generates data from pre-processed (smooth) curves

run(storageFile, sheetName) # "Do desired stuff" depends on what is written there, duh..


