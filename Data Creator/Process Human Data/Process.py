# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 17:43:49 2018

@author: Michael
"""

import random
import sys

letters = 'abcdefghijklmnopqrstuvwxyz'

#check the latest number that has been assigned
raw = open('HumanData.txt','r')
data = str(raw.read())
raw.close()

#update log
final = open('Processed_data.txt','w+')
for chars in data:
    if chars.lower() in letters:
        final.write(chars.lower())
final.close()