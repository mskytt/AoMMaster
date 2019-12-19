import h5py
import numpy as np
import os.path

"""
	Loading/reading to hdf5 file
	Create file and store data

	v0.1 - Mans Skytt (m@skytt.eu)
"""
def storeToHDF5(filename, datasetName, data_):
	if os.path.isfile(filename): # Pre-existing file, only append
		f = h5py.File(filename, 'a')
	else: # Not pre-existing, write
		f = h5py.File(filename, 'w')  
	try:
		f.create_dataset(datasetName, data = data_)
	except RuntimeError: # Name exists, replace!
		print 'Name already exists, replacing!'
		del f[datasetName]
		f.create_dataset(datasetName, data= data_)	
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



