import pandas as pd
from CompereAlgo import getsDataPaths,preprocess
import matplotlib.pyplot as plt
import matplotlib

def createWinAlgoHist():
    winAlgo = pd.read_csv("../data/results/winAlgo.csv")
    winAlgo = winAlgo[winAlgo["win"] == True]
    xx = winAlgo["AlgoName"].value_counts()
    ax = xx.plot.barh()
    ax.set_xlabel("win amount")
    
def classificationDatasetsInfo():
    
    params = {'axes.titlesize':'32',
          'xtick.labelsize':'24',
          'ytick.labelsize':'24'}
    matplotlib.rcParams.update(params)
    
    paths = getsDataPaths()
    rows_list = []
    for i,path in enumerate(paths):
        row = {}
        X, y = preprocess(path)
        
        datasetName = path.split("\\")[1].split(".")[0]
        row["name"] = datasetName
        
        row["size"] = len(X)
        
        row["labelAmount"] = len(y.unique())
        
        rows_list.append(row)
    allData = pd.DataFrame(rows_list)       

    allData.hist(bins = 50)   
    plt.tight_layout()
    plt.show()     
    return allData
        


df = classificationDatasetsInfo()





