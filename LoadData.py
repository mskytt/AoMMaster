
import os.path
import tools

class SpreadsheetData(object):

    def __init__(self, xlpath, projdir):

        self.xlpath = xlpath
        self.projdir = projdir
        if os.path.exists(xlpath):
            self.ws = pandas.read_excel(xlpath)
        else:
            raise FileNotFoundError(
                "File not found: " + xlpath)

        # as a convention, uppercase all columns
        # and strip trailing whitespace
        self.ws.columns = map(str.upper, self.ws.columns)
        self.ws.columns = map(str.rstrip, self.ws.columns)





tools.install_and_import('pickle')
tools.install_and_import('pandas')




v = SpreadsheetData('./Data/GoldFutures.xlsx', './Data')

