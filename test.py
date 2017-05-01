import numpy as np


A = [3, 10, 3, 5]
A = np.float(A)
logA = np.log(A)

print  A[0:-1]
print  A[1:]

print np.log(A[0:-1]) - np.log(A[1:])