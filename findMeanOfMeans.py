#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 12:00:15 2020

@author: Jaakko Ahola, Finnish Meteorological Institute
@licence: MIT licence Copyright
"""
import pandas
import numpy
import time



def main():
    design = pandas.read_csv("/home/aholaj/Data/EmulatorManuscriptData/LVL3Day/LVL3Day_design.csv")
    
    newnameList = []
    for colname in ["tpot_pbl", "pblh", "q_inv", "tpot_inv", "lwp"]:
        newname = "square_dist_from_mean_" + colname
        newnameList.append(newname)
        design[newname] = (design[colname] -design[colname].mean())/design[colname].std(ddof=1)
        #numpy.power( (design[colname]-design[colname].mean()) / (design[colname].max()),2)
    
    design["met.Mean"] = design["square_dist_from_mean_tpot_pbl"]*design["square_dist_from_mean_pblh"]*design["square_dist_from_mean_q_inv"]*design["square_dist_from_mean_tpot_inv"]*design["square_dist_from_mean_lwp"]
    
    print(design.sort_values(by="met.Mean").iloc[0])
    print((design.sort_values(by=["met.Mean", "pblh"]).head()))
          #"{0:d}".format                                                     
if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("Script completed in " + str(round((end - start),0)) + " seconds")
