# -*- coding: utf-8 -*-

'''
To use this code we need to take each new measuers to the path: "../data/backup" from code folder
and for each measuers we created we need to change the name to  measuersPart<i> where i is the number
for the append method.

After that we create new measure file and export to the result folder
'''

import pandas as pd

inputPath = "../data/backup"
outputPath = "../data/results"

allData = pd.DataFrame()
allDataPickle = pd.DataFrame()

for i in range(1,4):
     allData = allData.append(pd.read_csv(f"{inputPath}/measuersPart{i}.csv"))
     
allData.drop_duplicates(subset = ["AlgoName","Dataset_Name","CrossVal"],inplace=True)


allData.to_csv(f"{outputPath}/measuers.csv")