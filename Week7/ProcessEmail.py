# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 09:10:27 2015

@author: Charleseagle
"""
import re
import nltk, nltk.stem.porter
import GetVocabList


def ProcessEmail(email_contents = None):
    vocablist = GetVocabList.GetVocabList()
    
    word_indices = []
    email_contents = email_contents.lower()
    # Strip all HTML
    #Looks for any expression that starts with < and ends with > and replace
    # and does not have any < or > in the tag it with a space 
    

    email_contents = re.sub('<[^<>]+>', ' ', email_contents)
    # Look for one or more characters between 0-9
    email_contents = re.sub('[0-9]+', 'number', email_contents)
    # Look for strings starting with http:// or https://
    email_contents = re.sub('(http|https)://[^\s]*', 'httpaddr', email_contents)
    # Look for strings with @ in the middle
    email_contents = re.sub('[^\s]+@[^\s]+', 'emailaddr', email_contents)
    # Handle $ sign
    email_contents = re.sub('[$]+', 'dollar', email_contents) 
    
    print '\n==== Processed Email ===='
    
    if email_contents != None:
    # Tokenize and also get rid of any punctuation   
        stemmer = nltk.stem.porter.PorterStemmer()
        tokens = re.split('[ ' + re.escape("@$/#.-:&*+=[]?!(){},'\">_<;%") + ']' ,\
                email_contents )
        for token in tokens:
            token = re.sub('[^a-zA-Z0-9]', '', token)
            token = stemmer.stem(token.strip())
            if len(token) == 0:
                continue
    
            if token in vocablist:
                word_indices.append(vocablist[token] )
    return word_indices  
    
    
    
    
    
    
    