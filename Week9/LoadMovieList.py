# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 23:51:50 2015

@author: Charleseagle
"""



def LoadMovieList():
    counter = 0
    movielist = {}
    with open('E:\Machine learning\Week_9\Python\movie_ids.txt', 'rb') as fid:
        lines = fid.readlines()
        for line in lines:
            movielist[counter] = line.split(' ', 1)[1]
            counter += 1
    return movielist
