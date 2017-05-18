from forwardCurves import runGenerateData, runGenZCPCs, runGenForPCs, runGenMatlab, run
from h5pyStorage import loadFromHDF5

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
MATLABstorageFile = 'MatlabEONIA05midForward100.hdf5' 
"""
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
# runGenerateData(readExcel, genForward, genZC, sheetName, storageFile)
# ZCMatDiff = loadFromHDF5(storageFile,'ZCMatDiff')
# runGenZCPCs(genZCEigs, ZCMatDiff, storageFile)
# forMatDiff = loadFromHDF5(storageFile,'forMatDiff')
# runGenForPCs(genForEigs, forMatDiff, storageFile)
MATLABForwardMat = loadFromHDF5(MATLABstorageFile,'MATLABFordataMat')
runGenMatlab(genMatlab, genMatlabEigs, MATLABForwardMat, sheetName, storageFile)
run(storageFile)


