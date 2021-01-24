# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 16:08:08 2020

@author: oxenb
"""

import gzip
import shutil

path_from = 'tagger.py'
path_to = 'ex4.py.gz'

with open(path_from, 'rb') as f_in:
    with gzip.open(path_to, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)