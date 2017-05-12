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

"""
    Run functions
"""
runGenerateData(readExcel, genForward, genZC)
runGenZCPCs(genZCEigs)
runGenForPCs(genForEigs)
runGenMatlab(genMatlab, genMatlabEigs)
# run()


