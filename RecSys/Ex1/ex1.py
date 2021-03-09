#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 10:37:08 2021

@author: saharbaribi
"""

import pandas as pd 
import time




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

start_time = time.time()
Load()
end_time = time.time()
print(f"time to load data:{end_time - start_time}")

