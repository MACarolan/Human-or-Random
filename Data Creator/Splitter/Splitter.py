# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 17:43:49 2018

@author: Michael
"""

import random
import sys

"""
INPUT name of files to process and desired length of substrings
"""

letters = 'abcdefghijklmnopqrstuvwxyz'

f_name = sys.argv[1]
length = int(sys.argv[2])

#retrieve the data from the desired file
raw = open(f'{f_name}.txt','r')
data = raw.read()
raw.close()

#create samples
new_file = open(f'human_data_len{length}.txt','w+')
for starts in range(0, len(data), length//2): #duplicate middle sections
    
    #make sequences of given length
    if starts+length < len(data): # len to make must be less than whats left
        new_file.write(data[starts:starts+length] + '\n')
        
new_file.close()