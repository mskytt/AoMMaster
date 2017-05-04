import h5py
import numpy as np
import os.path

"""
	Create file and store data
"""
def storeToHDF5(filename, datasetName, data_):
	if os.path.isfile(filename): # Pre-existing file, only append
		f = h5py.File(filename, 'a')
	else: # Not pre-existing, write
		f = h5py.File(filename, 'w')  
	
	f.create_dataset(datasetName, data = data_)
	f.close()
	return
"""
	Load data from file
"""
def loadFromHDF5(filename, datasetName):
	try:
		f = h5py.File(filename, 'r')
		loadedData = f[datasetName][:]
	except IOError:
		print "No file called: " + filename
		return None
	except KeyError:
		print "No dataset called: " + datasetName
		return None
	f.close()
	return loadedData

# a = np.array([9, 9, 9, 67, 7])
# aLoaded = 0
# storeToHDF5('EONIAask.hdf5', 'OISdataMat', a)
# storeToHDF5('ArrayStorage.hdf5','aMat',a)
# aLoaded = loadFromHDF5('EONIAask.hdf5','OISdataMat')

# print aLoaded



