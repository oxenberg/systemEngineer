#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 10:37:08 2021

@author: saharbaribi
"""

import pandas as pd 
import time
from sqlalchemy import create_engine
import os.path
from os import path 


class Database():
    def __init__(self, write_to_database = False):
        self.files = {'userTrainData':"Unnamed: 0",
                      'userTestData':"Unnamed: 0", 
                      'yelp_business':"buisness_id", 
                      'yelp_user': "user_id"}
        self.yelp_database = create_engine('sqlite:///yelp_database.db')
        self.write_to_database(write_to_database)
    
    def write_to_database(self,write_to_database):
        
        if write_to_database:
            for file in self.files: 
                self.create_table(file)
    
    def create_table(self, file):
        file_name = file+'.csv'
        chunksize = 100000
        i = 0
        j = 1
        #: For trainData and testData we want to remove the index column. 
        for df in pd.read_csv(file_name, chunksize=chunksize, iterator=True, index_col = self.files[file]):
              df = df.rename(columns={c: c.replace(' ', '') for c in df.columns})
              df.index += j
              i+=1
              df.to_sql(file, self.yelp_database, if_exists='append')
              j = df.index[-1] + 1        
              
              
    def query_database(self, query):
        df = pd.read_sql_query(query, self.yelp_database)
        return df 

              
  



def Load():
    write_to_database = True
    if path.exists("yelp_database.db"):
        write_to_database = False 
    DB = Database(write_to_database)
    return DB


DB = Load()
query = 'SELECT business_id, AVG(stars) as avg_stars FROM userTrainData GROUP BY business_id'
df = DB.query_database(query)


# start_time = time.time()
# Load()
# end_time = time.time()
# print(f"time to load data:{end_time - start_time}")

