#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 10:37:08 2021

@author: saharbaribi
"""

import pandas as pd 
import time
from sqlalchemy import create_engine




def Load():
    trainData = pd.read_csv("userTrainData.csv", index_col="Unnamed: 0", dtype={"user_id": str, 
                                                                            "business_id": str, 
                                                                            "stars": int, 
                                                                            "text":str})
    testData = pd.read_csv("userTestData.csv", index_col="Unnamed: 0", dtype={"user_id": str, 
                                                                            "business_id": str, 
                                                                            "stars": int, 
                                                                            "text":str})
    buisnessData = pd.read_csv("yelp_business.csv")    
    userData = pd.read_csv("yelp_user.csv")
    
    return trainData, testData, userData, buisnessData

file  = "userTrainData.csv"
csv_database = create_engine('sqlite:///csv_database.db')
chunksize = 100000
i = 0
j = 1
for df in pd.read_csv(file, chunksize=chunksize, iterator=True):
      df = df.rename(columns={c: c.replace(' ', '') for c in df.columns})
      df.index += j
      i+=1
      df.to_sql('table', csv_database, if_exists='append')
      j = df.index[-1] + 1


df = pd.read_sql_query('SELECT * FROM table', csv_database)


# start_time = time.time()
# Load()
# end_time = time.time()
# print(f"time to load data:{end_time - start_time}")

