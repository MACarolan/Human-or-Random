# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 15:19:02 2018

@author: Michael
"""

import random
import sys

"""
INPUT number of files desired, desired number of strings, and desired length
"""

letters = 'abcdefghijklmnopqrstuvwxyz'

if sys.argv[1] != 'r':
    #check the latest number that has been assigned
    log = open('LOG.txt','r')
    latest = int(log.read()[0])
    log.close()
    
    size = int(sys.argv[2])
    length = int(sys.argv[3])
    
    for files in range(int(sys.argv[1])):
        new_file = open(f'rand_data_{latest+1}.txt','w+')
        
        #make 1000 10-letter random sequences
        for data in range(size):
            new_file.write(''.join(random.choices(letters,k=length)) + '\n')
            
        new_file.close()
        latest += 1

else:
    latest = 0

#update log
log = open('LOG.txt','w+')
log.write(f'{latest}')
log.close()

