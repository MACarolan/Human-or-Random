# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 15:19:02 2018

@author: Michael
"""

import random
import sys

letters = 'abcdefghijklmnopqrstuvwxyz'

#check the latest number that has been assigned
log = open('LOG.txt','r')
latest = int(log.read()[0])
log.close()

for files in range(int(sys.argv[1])):
    new_file = open(f'rand_data_{latest+1}.txt','w+')
    
    #make 1000 10-letter random sequences
    for data in range(1000):
        new_file.write(''.join(random.choices(letters,k=10)) + '\n')
        
    new_file.close()
    latest += 1

#update log
log = open('LOG.txt','w+')
log.write(f'{latest}')
log.close()

