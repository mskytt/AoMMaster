#this is a draft for a correlation analysis model

#import packages
import tools
import_modules = ['scipy.stat', 'math', 'numpy','arch', 'datetime']

#simple calc methods

#TODO: change to vector?
def calculate_log_returns(matrix):

    log_returnMatrix =  math.log((matrix[0:-2,:]/matrix[1:-1,:]) -1)

    return log_returnsMatrix



#TODO: should we do this on vectors or matrices?
def main_test(data):

    for timeseries in data:
    log_returns = calculate_log_returns(timeseries)
    if archInstallSucessfull:
        sigmas = []
        sigmas.append(getGARCH11Volatilities(log_returns))
        #DCC


# ---------------GARCH ---------------

#def getGARCH11Volatilities():



def garch11(data):

    model = arch.arch_model(data, p=1, q=1)
    res = model.fit(update_freq=10)
   #TODO
    fig = res.plot(annualize='D')
    print(res.summary())




print("test")


for module in import_modules:
    try:
        tools.install_and_import(module)
    except ImportError:
        tools.archInstallSucessfull(False)
        print module + " not imported"




