#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 15:46:01 2020

@author: Jaakko Ahola, Finnish Meteorological Institute
@licence: MIT licence Copyright
"""
import matplotlib
import numpy
import pandas
import seaborn
import time

from Colorful import Colorful
from Data import Data
from InputSimulation import InputSimulation
from Figure import Figure
from Plot import Plot
from PlotTweak import PlotTweak


def mergeDataFrameWithParam(simulationDataFrames, paramDict, paramName):
    for ind, case in enumerate(list(simulationDataFrames)):
        paramDataFrame  = pandas.DataFrame({"ID":list(paramDict[case]),
                              paramName : list(paramDict[case].values())})
        del simulationDataFrames[case]["ID"]
        simulationDataFrames[case] =  pandas.merge( simulationDataFrames[case],
                                                        paramDataFrame, on ="ID" )
    
    return simulationDataFrames


def outliersFromDataFrame(dataframe : pandas.DataFrame , variable : str, outlierFrac : float):
    
    low = dataframe[variable].quantile( outlierFrac )
    high  = dataframe[variable].quantile(1.-outlierFrac)
    
    if high < low:
        apu = low
        low = high
        high = apu
        
    
    
    return dataframe[( dataframe[variable] < low ) | (dataframe[variable] > high  )]



def midQuantileFromDataFrame(dataframe : pandas.DataFrame , variable : str, outlierFrac : float): #variable  < 0.5
    
    low = dataframe[variable].quantile( outlierFrac )
    high  = dataframe[variable].quantile(1.-outlierFrac)
    
    if high < low:
        apu = low
        low = high
        high = apu

    
    return dataframe[( dataframe[variable] >  low ) & ( dataframe[variable] < high  )]
    
def plot4Sets(caseCollection, simulationCollection, annotationCollection, simulationDataFrames,
              figurefolder, figurename,
              ncVariable, designVariable,
              conversionNC = 1.0, conversionDesign = 1.0,
              xmax = 1000, ymax = 1000,
              xAxisLabel  = None, xAxisUnit = None,
              yAxisLabel = None, yAxisUnit = None, keisseja = 10000,
              yPositionCorrection = 100, outlierParameter = 0.2):
    
    relativeChangeDict  = Data.emptyDictionaryWithKeys(caseCollection)
    print(" ")
    print(figurename)
    # create figure object
    fig = Figure(figurefolder,figurename, ncols=2, nrows=2, sharex=True, sharey = True)
    # plot timeseries with unit conversion
    maks = 0
    mini = 0
    for ind, case in enumerate(caseCollection):
        for emul in list(simulationCollection[case])[:keisseja]:
            dataset = simulationCollection[case][emul].getTSDataset()
            muuttuja = dataset[ncVariable]
            
            alku = simulationDataFrames[case].loc[emul][designVariable] * conversionDesign
            loppu = muuttuja.sel(time=slice(2.5, 3.5)).mean().values * conversionNC
            relChangeParam = loppu/alku
            
            
            relativeChangeDict[case][emul] = relChangeParam
            
            if relChangeParam > 1 + outlierParameter:
                color = Colorful.getDistinctColorList("red")
                zorderParam = 10
            elif relChangeParam < 1- outlierParameter:
                color = Colorful.getDistinctColorList("blue")
                zorderParam = 9
            else:
                color = "white"
                zorderParam  = 6
            
                    
                    
            maks = max(relChangeParam, maks)
            mini = min(relChangeParam, mini)
            
            fig.getAxes(True)[ind].plot( alku,  loppu,
                                             marker = "o",
                                             markerfacecolor = color,
                                             markeredgecolor = "black",
                                             markeredgewidth=0.2,
                                             markersize = 6,
                                             alpha = 0.5,
                                             zorder=zorderParam
                                             )
    print("minimi", mini, "maksimi",maks)
        
    for ind, case in enumerate(caseCollection):
        PlotTweak.setAnnotation(fig.getAxes(True)[ind], annotationCollection[case], xPosition=100, yPosition=ymax-yPositionCorrection)
        PlotTweak.setXLim(fig.getAxes(True)[ind],0,xmax)
        PlotTweak.setYLim(fig.getAxes(True)[ind],0,ymax)
        fig.getAxes(True)[ind].plot( [0,xmax], [0, ymax], 'k-', alpha=0.75, zorder=0)

    PlotTweak.setXaxisLabel( fig.getAxes(True)[2], xAxisLabel, xAxisUnit, useBold=True)
    PlotTweak.setXaxisLabel( fig.getAxes(True)[3], xAxisLabel, xAxisUnit, useBold=True)
    PlotTweak.setYaxisLabel( fig.getAxes(True)[0], yAxisLabel, yAxisUnit, useBold=True)
    PlotTweak.setYaxisLabel( fig.getAxes(True)[2], yAxisLabel, yAxisUnit, useBold=True)
    PlotTweak.setLegend(fig.getAxes(True)[0], {"Change > +20%" : Colorful.getDistinctColorList("red"),
                                                "Change < 20% " : "white",
                                                "Change < -20%" : Colorful.getDistinctColorList("blue"),
                                                    }, loc=(0.02,0.6), fontsize = 6)
    return fig, relativeChangeDict

def main():
    
    caseCollection = ["LVL3Night",
                "LVL3Day",
                "LVL4Night",
                "LVL4Day"]
    
    annotationValues = ["(a) LVL3 Night",
            "(b) LVL3 Day",
            "(c) LVL4 Night",
            "(d) LVL4 Day"]
    
    annotationCollection = dict(zip(caseCollection, annotationValues))
    
    csvFolder = "/home/aholaj/OneDrive/000_WORK/000_ARTIKKELIT/001_Manuscript_LES_emulator/data/"
    
    simulationDataFrames = {}
    for case in caseCollection:
        simulationDataFrames[case] = pandas.read_csv( csvFolder + case + ".csv")
        simulationDataFrames[case] = simulationDataFrames[case].set_index("ID", drop = False)
        
    
    simulationCollection ={}
    for case in caseCollection:
        simulationCollection[case] = InputSimulation.getSimulationCollection( 
                                                         simulationDataFrames[case] )
    
    #figure folder
    figurefolder = "/home/aholaj/OneDrive/000_WORK/000_ARTIKKELIT/001_Manuscript_LES_emulator/figures"
    cloudTopOutliers = {}
    
    # load ts-datasets and change their time coordinates to hours
    keisseja = 10000  # TODO REMOVE THIS
    for case in caseCollection:
        for emul in list(simulationCollection[case])[:keisseja]:
            print(emul)
            
            simulationCollection[case][emul].getTSDataset()
            #simulationCollection[case][emul].getPSDataset()
            simulationCollection[case][emul].setTimeCoordToHours()
    
    
    LWPFigureFlag = True
    CloudTopFigureFlag = True
    cloudTopOutliersFlag = True
    LWPOutliersFlag = True
    if LWPFigureFlag:
        lwpFig, lwpChangeParameters = plot4Sets(caseCollection, simulationCollection, annotationCollection, simulationDataFrames, figurefolder, "lwp",
              "lwp_bar", "lwp",
              conversionNC = 1000., conversionDesign = 1.0,
              xmax = 1000, ymax = 1000,
              xAxisLabel = "LWP", xAxisUnit = "g m^{-2}",
              yAxisLabel = "LWP", yAxisUnit = "g m^{-2}", keisseja = keisseja, outlierParameter=0.5)
        
        simulationDataFrames = mergeDataFrameWithParam( simulationDataFrames, lwpChangeParameters, "lwpRel")
        
        lwpFig.save()
        
    if CloudTopFigureFlag:
        cloudTopFig, cloudTopParameters = plot4Sets(caseCollection, simulationCollection, annotationCollection, simulationDataFrames, figurefolder, "cloudtop",
              "zc", "pblh_m",
              xmax = 3600, ymax = 3600,
              xAxisLabel = "Cloud\ top", xAxisUnit = "m",
              yAxisLabel = "Cloud\ top", yAxisUnit = "m", keisseja = keisseja,
              yPositionCorrection=300)
        simulationDataFrames = mergeDataFrameWithParam( simulationDataFrames, cloudTopParameters, "zcRel")
        cloudTopFig.save()
    
    if cloudTopOutliersFlag:
        cloutTopOutliers = {}
        for ind, case in enumerate(list(simulationDataFrames)):
            cloudTopOutliers[case] =  simulationDataFrames[case].where( simulationDataFrames[case]["zcRel"] <  simulationDataFrames[case]["zcRel"] )
        

        fig2 = Figure(figurefolder,"cloudtopOutliers", ncols=2, nrows=2, sharex=True, sharey = True)
        # plot timeseries with unit conversion
        cloudTopOutliersColors = Colorful.getIndyColorList(len(cloudTopOutliers))
        for ind, case in enumerate(caseCollection):
            for emulInd, emul in enumerate(cloudTopOutliers):
                try:
                    simulation = simulationCollection[case][emul]
                except KeyError:
                    continue
                dataset = simulation.getTSDataset()
                muuttuja = dataset["zc"][1:] / ( simulationDataFrames[case].loc[emul]["pblh_m"])
                
                muuttuja.plot( ax = fig2.getAxes(True)[ind],
                              color = cloudTopOutliersColors[emulInd],
                              label =  simulationCollection[case][emul].getLabel() ) 
                
            
        xmax = 3.5
        ymax = 2.5
        
        for ind, case in enumerate(caseCollection):
            PlotTweak.setAnnotation(fig2.getAxes(True)[ind], annotationCollection[case], xPosition=1.5, yPosition=ymax-0.25)
            PlotTweak.setXLim(fig2.getAxes(True)[ind],0,xmax)
            PlotTweak.setYLim(fig2.getAxes(True)[ind],0,ymax)
            
            #fig2.getAxes(True)[ind].plot( [0,xmax], [0, ymax], 'k-', alpha=0.75, zorder=0)
            
            PlotTweak.useLegend(fig2.getAxes(True)[ind], loc = 'upper left')
            PlotTweak.setXaxisLabel( fig2.getAxes(True)[ind], "", None, useBold=True)
            PlotTweak.setYaxisLabel( fig2.getAxes(True)[ind], "", None, useBold=True)
            Plot.getVerticalLine( fig2.getAxes(True)[ind], 1.5)
        
        
        PlotTweak.setXaxisLabel( fig2.getAxes(True)[2], "Time", "h", useBold=True)
        PlotTweak.setXaxisLabel( fig2.getAxes(True)[3], "Time", "h", useBold=True)
        PlotTweak.setYaxisLabel( fig2.getAxes(True)[0], "Cloud top relative change", None, useBold=True)
        PlotTweak.setYaxisLabel( fig2.getAxes(True)[2], "Cloud top relative change", None, useBold=True)
        PlotTweak.setYaxisLabel( fig2.getAxes(True)[2], "Cloud top relative change", None, useBold=True)
        PlotTweak.setYaxisLabel( fig2.getAxes(True)[2], "Cloud top relative change", None, useBold=True)
        
        
        fig2.save()        

    if LWPOutliersFlag:
        for ind, case in enumerate(list(lwpOutliers)):
            lwpOutliers[case] = Data.getHighAndLowTail(lwpOutliers[case], 0.01)
        fig2 = Figure(figurefolder,"lwpOutliers", ncols=2, nrows=2, sharex=True, sharey = True)
        # plot timeseries with unit conversion
        lwpOutliersColors = Colorful.getIndyColorList(len(lwpOutliers))
        for ind, case in enumerate(list(lwpOutliers)):
            for emulInd, emul in enumerate(list(lwpOutliers[case])):
                try:
                    simulation = simulationCollection[case][emul]
                except KeyError:
                    continue
                dataset = simulation.getTSDataset()
                muuttuja = dataset["lwp_bar"]*1000. / ( simulationDataFrames[case].loc[emul]["lwp"])
                
                muuttuja.plot( ax = fig2.getAxes(True)[ind],
                              color = lwpOutliersColors[emulInd],
                              label =  simulationCollection[case][emul].getLabel() ) 
                
            
        xmax = 3.5
        ymax = 2.5
        
        for ind, case in enumerate(caseCollection):
            PlotTweak.setAnnotation(fig2.getAxes(True)[ind], annotationCollection[case], xPosition=1.5, yPosition=ymax-0.25)
            PlotTweak.setXLim(fig2.getAxes(True)[ind],0,xmax)
            PlotTweak.setYLim(fig2.getAxes(True)[ind],0,ymax)
            
            #fig2.getAxes(True)[ind].plot( [0,xmax], [0, ymax], 'k-', alpha=0.75, zorder=0)
            
            PlotTweak.useLegend(fig2.getAxes(True)[ind], loc = 'upper left')
            PlotTweak.setXaxisLabel( fig2.getAxes(True)[ind], "", None, useBold=True)
            PlotTweak.setYaxisLabel( fig2.getAxes(True)[ind], "", None, useBold=True)
            Plot.getVerticalLine( fig2.getAxes(True)[ind], 1.5)
        
        
        PlotTweak.setXaxisLabel( fig2.getAxes(True)[2], "Time", "h", useBold=True)
        PlotTweak.setXaxisLabel( fig2.getAxes(True)[3], "Time", "h", useBold=True)
        PlotTweak.setYaxisLabel( fig2.getAxes(True)[0], "LWP relative change", None, useBold=True)
        PlotTweak.setYaxisLabel( fig2.getAxes(True)[2], "LWP relative change", None, useBold=True)
        PlotTweak.setYaxisLabel( fig2.getAxes(True)[2], "LWP relative change", None, useBold=True)
        PlotTweak.setYaxisLabel( fig2.getAxes(True)[2], "LWP relative change", None, useBold=True)
        
        
        fig2.save()         
        
        
#    Plot.getTimeseries(fig2.getAxes()[0,0],
#                       list( simulationCollection["LVL3Night"].values())[:10],
#                       "lwp_bar", conversionFactor=1000.)
#    Plot.getVerticalLine(fig2.getAxes()[0,0], 1.5)
#    
#    if True:
#        PlotTweak.setSuperWrapper(fig2.getAxes()[0,0],
#                              xstart = 0,
#                              xend = 3.5,
#                              xtickinterval = 0.5,
#                              xlabelinterval = 1,
        
#                              xticks = None,
#                              xShownLabels = None,
#                              xTickMajorFontSize = 3.5, # matplotlib.rcParams["xtick.major.size"],
#                              yTickMajorFontSize = 3.5, #matplotlib.rcParams["ytick.major.size"],
#                              ystart = 0, 
#                              yend = 800,
#                              ytickinterval = 50,
#                              ylabelinterval = 100,
#                              yticks = None,
#                              yShownLabels = None,
#                              annotationText = None,
#                              annotationXPosition = 2,
#                              annotationYPosition = 30,
#                              xlabel = "Time",
#                              xunit  = "h",
#                              ylabel = "LWP",
#                              yunit  = "g\ m^{-2}",
#                              useBold = True
#                              )
#    PlotTweak.hideYTickLabels( fig2.getAxes()[0,0] )
#    
#    PlotTweak.setAxesOff( fig2.getAxes()[0,1] )
#    PlotTweak.setLegendSimulation(fig2.getAxes()[0,1], [simulationCollection["ICE0_8h"], simulationCollection["ICE1_8h"]])
#    
#    ax, im, levels = Plot.getTimeseriesOfProfile(fig2.getAxes()[1,1],
#                     simulationCollection["ICE0_8h"],
#                     "P_RH",
#                     levels = None,
#                     useLogaritmic = False,
#                     colors = None)
#    PlotTweak.setAxesOff( fig2.getAxes()[1,0] )
#    cax = matplotlib.pyplot.axes([0.05, 0.25, 0.3, 0.03])
#    Plot.getColorBar(im, cax, levels, pad = 0.22)
    
    #Plot.getContourLine(fig2.getAxes()[1,1],"ICE0_8h", color =  'black' , value = 100)
    
    

        
if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("Script completed in " + str(round((end - start),0)) + " seconds")
