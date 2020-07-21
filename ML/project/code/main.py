# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 10:44:38 2020

@author: oxenb
"""

from CompereAlgo import CompereAlgo
from Statistics import calcStatistics
from Metaclassifiar import runMetaclassifier

def main():
    """main for project, configure set parmeters for training test valdiation CV and
       random search iterations
        
        ----------
                
        Returns
        -------
            
    """
    testTrainCV = 2
    trainValCV = 2
    randomSearchIter = 2
    CompereAlgo(testTrainCV = testTrainCV,trainValCV = trainValCV,randomSearchIter = randomSearchIter)
    
    statisticsResult = calcStatistics()
    
    runMetaclassifier()
    







if __name__ == "__main__":
    main()
    
