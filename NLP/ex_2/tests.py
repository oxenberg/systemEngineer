# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 15:52:21 2020

@author: oxenb
"""

import ex2

table = ex2.EditDist("#xoen","#oxen")
table.fillTable()


ta = table.distTable
x = table.getReversedPath()

# for i in x:
#     print(i)


t = table.getOpeartions()
print(t)


# sp = ex2.Spell_Checker()
# d = sp.learn_error_tables("commmon_errors.txt")


    
# count = 0
# for value in d.values():
#     count+=len(value.values())
