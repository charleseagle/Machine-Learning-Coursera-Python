# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 09:11:19 2015

@author: Charleseagle
"""
from __future__ import division
import csv



def GetVocabList():
    
    vocablist = {}
    with open('E:/Machine learning/Week_7/Python/vocab.txt', 'r') as fid:
        text = csv.reader(fid, delimiter='\t' )
        for row in text:
            vocablist[row[1]] = int(row[0])
            
    return vocablist