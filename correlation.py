
#this is a draft for a correlation analysis model

#edit 1 Anna


#import packages

import stat
import math
import numpy as np
import random
#from arch import arch_model

import sys
print '\n'.join(sys.path)
np.__file__

import imp
imp.find_module('arch')
#class timeseries(self):



#create time series



#simple calc methods

def calculate_log_returns(timeseries):

    log_returns =  math.log((timeseries[-1]/timeseries[1:]) -1)

    return log_returns


def garch11(data):

    model = arch_model(data, p=1, q=1)
    res = model.fit(update_freq=10)
    print(res.summary())

def main_test(timeseries):
    data = calculate_log_returns(timeseries)

    garch11(data)



timeseries = np.array(random.random((100,)))

print("test")

main_test(timeseries)


