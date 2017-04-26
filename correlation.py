
#this is a draft for a correlation analysis model

#edit 1 Anna


#import packages
from scipy import stat
import math
import numpy as np
import random
#from arch import arch_model

import FormatData

#simple calc methods

def calculate_log_returns(matrix):

    log_returnMatrix =  math.log((matrix[0:-2,:]/matrix[1:-1,:]) -1)

    return log_returnsMatrix




def main_test(timeseries):
    data = calculate_log_returns(timeseries)

    garch11(data)



# ---------------GARCH ---------------

# def garch11(data):
#
#     model = arch.arch_model(data, p=1, q=1)
#     res = model.fit(update_freq=10)
#     print(res.summary())

#def getGARCH11Volatilities():




print("test")

tools.install_and_import('arch')


