from forwardCurves import runGenerateData, runGenZCPCs, runGenForPCs, run
"""
    Set bools to choose where to get data from
"""
readExcel = False # Read from excel
genForward = False # Generate forward rates matrix
genZC = False # Generate zero coupon rates matrix
genForEigs = False # Generate forward eigenvalues
genZCEigs = False # Generate zero coupon eigenvalues

"""
    Run functions
"""
runGenerateData(readExcel, genForward, genZC)
runGenZCPCs(genZCEigs)
runGenForPCs(genForEigs)
run()