#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 15:46:01 2020

@author: Jaakko Ahola, Finnish Meteorological Institute
@licence: MIT licence Copyright
"""
import matplotlib
import math
import numpy
import os
import pandas
import pathlib
import seaborn
import scipy
import sys
import time

sys.path.append("../LES-03plotting")
from Colorful import Colorful
from Data import Data
from InputSimulation import InputSimulation
from Figure import Figure
from Plot import Plot
from PlotTweak import PlotTweak

from sklearn.metrics import mean_squared_error
from math import sqrt

class ManuscriptFigures:
    
    def __init__(self, emulatorPostprosDataRootFolder, figurefolder):
        
        self.trainingSetList = ["LVL3Night",
                               "LVL3Day",
                               "LVL4Night",
                               "LVL4Day"]

        self.emulatorPostprosDataRootFolder = pathlib.Path(emulatorPostprosDataRootFolder)
        self.figurefolder = pathlib.Path(figurefolder)
        
        self.simulationCollection = {}
        self.simulationDataFrame = {}
        for trainingSet in self.trainingSetList:
            simulationDataFrame = pandas.read_csv( self.emulatorPostprosDataRootFolder / ( trainingSet + ".csv" )  )
            simulationDataFrame = simulationDataFrame.set_index("ID", drop = False)
            self.simulationDataFrame[trainingSet] = simulationDataFrame
            self.simulationCollection[trainingSet] = InputSimulation.getSimulationCollection( simulationDataFrame )
        
        self.annotationValues = ["(a) SB Night",
            "(b) SB Day",
            "(c) SALSA Night",
            "(d) SALSA Day"]

        self.annotationCollection = dict(zip(self.trainingSetList, self.annotationValues))
        
        self.cloudTopColor = Colorful.getDistinctColorList("green")
        self.lwpColor = Colorful.getDistinctColorList("blue")
        self.tempColor = Colorful.getDistinctColorList("yellow")
        
        self.bootStrapSampleSize = 100
        self.bootStrapNumberOfSamples = 10000
    def readSimulatedVSPredictedData(self):
        self.simulatedVSPredictedCollection = {}
        for trainingSet in self.trainingSetList:
            self.simulatedVSPredictedCollection[trainingSet] = pandas.read_csv( self.emulatorPostprosDataRootFolder / trainingSet / ( trainingSet + "_simulatedVSPredictedData.csv" ) )

    def readSensitivityData(self):
        self.sensitivityDataCollection = {}
        for trainingSet in self.trainingSetList:
            self.sensitivityDataCollection[trainingSet] = pandas.read_csv( self.emulatorPostprosDataRootFolder / trainingSet / ( trainingSet + "_sensitivityAnalysis.csv" ) )
    
    def readResponseData(self):
        self.responseDataCollection = {}
        for trainingSet in self.trainingSetList:
            self.responseDataCollection[trainingSet] = pandas.read_csv( self.emulatorPostprosDataRootFolder / trainingSet / ( trainingSet + "_responseFromTrainingSimulations.csv" ) )
        

    def loadTimeSeriesLESData(self):
        # load ts-datasets and change their time coordinates to hours
        keisseja = 10  # TODO REMOVE THIS
        for trainingSet in self.trainingSetList:
            for emul in list(self.simulationCollection[trainingSet])[:keisseja]:
                
                self.simulationCollection[trainingSet][emul].getTSDataset()
                self.simulationCollection[trainingSet][emul].setTimeCoordToHours()
                
    def loadProfileLESData(self):
        # load ts-datasets and change their time coordinates to hours
        keisseja = 10  # TODO REMOVE THIS
        for trainingSet in self.trainingSetList:
            for emul in list(self.simulationCollection[trainingSet])[:keisseja]:
                self.simulationCollection[trainingSet][emul].getPSDataset()
                self.simulationCollection[trainingSet][emul].setTimeCoordToHours()
                
                
                
    def fillUpDrflxValues(self):
        
        for ind, trainingSet in enumerate(self.trainingSetList[-2:]):
            print(" ")
            responseData = self.responseDataCollection[trainingSet]
            
            if numpy.abs(responseData["drflx"].max() - responseData["drflx"].min() ) > 10*Data.getEpsilon(): 
                continue # drflx values already calculated
            else:
                print("let us refill missing drflx values", trainingSet)
            
            newCloudRadiativeValues = numpy.zeros(numpy.shape(responseData["drflx"]))
            
            for emulInd, emul in enumerate(list(self.simulationCollection[trainingSet])):
                self.simulationCollection[trainingSet][emul].getPSDataset()
                self.simulationCollection[trainingSet][emul].setTimeCoordToHours()
                
                tstart = 2.5
                tend = 3.5
                tol_clw=1e-5
                
                psDataTimeSliced = self.simulationCollection[trainingSet][emul].sliceByTimePSDataset(tstart, tend)
                
                numberOfCloudyColumns = 0
                
                cloudRadiativeWarmingAllColumnValues = 0.
                
                for timeInd, timeValue in enumerate(psDataTimeSliced["time"]):
                    rflxTimeSlice = psDataTimeSliced["rflx"].isel( time = timeInd )
                    
                    liquidWaterTimeSlice = psDataTimeSliced["l"].isel( time = timeInd )
                    
                    psDataCloudyPointIndexes, = numpy.where( liquidWaterTimeSlice > tol_clw )
                    # print(psDataCloudyPointIndexes)
                    if len(psDataCloudyPointIndexes > 0):
                        numberOfCloudyColumns += 1
                        
                        firstCloudyGridCell = psDataCloudyPointIndexes[0]
                        lastCloudyGridCell = psDataCloudyPointIndexes[-1]   
                        
                        #print("first and last", firstCloudyGridCell, lastCloudyGridCell)
                    
                        cloudRadiativeWarmingAllColumnValues += rflxTimeSlice[firstCloudyGridCell] - rflxTimeSlice[lastCloudyGridCell]
                        
                ## end time for loop
                if numberOfCloudyColumns > 0:
                    drflx = cloudRadiativeWarmingAllColumnValues.values / numberOfCloudyColumns
                else:
                    drflx = 0.
                
                newCloudRadiativeValues[emulInd] = drflx
                
            ### end emul for loop
            responseData["drflx"] = newCloudRadiativeValues
            
            print(responseData["drflx"])
            
            responseData.to_csv( self.emulatorPostprosDataRootFolder / trainingSet / ( trainingSet + "_responseFromTrainingSimulations.csv" ), index=False )
            
            
                
    def getBootstrapLinRegress(self, dataframe, xName, yName):
        bootStrapRSquared = numpy.zeros(self.bootStrapNumberOfSamples)            
        for state in range(self.bootStrapNumberOfSamples):
            dataframeSample = dataframe.sample( n=self.bootStrapSampleSize, random_state=state )
            slopeSample, interceptSample, r_valueSample, p_valueSample, std_errSample = scipy.stats.linregress(dataframeSample[xName], dataframeSample[yName])            
            bootStrapRSquared[state] = numpy.power(r_valueSample,2)
        
        return numpy.mean(bootStrapRSquared), numpy.std(bootStrapRSquared)
    
    def getBootstrapMeanAverage(self, dataframe, xName, yName):
        meanAbsErrorList = numpy.zeros(self.bootStrapNumberOfSamples)            
        for state in range(self.bootStrapNumberOfSamples):
            dataframeSample = dataframe.sample( n=self.bootStrapSampleSize, random_state=state )
            meanAbsError = numpy.mean(numpy.abs( dataframeSample[xName] - dataframeSample[yName] ) )
            meanAbsErrorList[state] = meanAbsError
        
        return numpy.mean(meanAbsErrorList), numpy.std(meanAbsErrorList)
    
    def getOutlierDataFromLESoutput(self):
    
        for ind, trainingSet in enumerate(self.trainingSetList):
            dataframe = self.simulationDataFrame[trainingSet]
                
            if ("lwpRelativeChange" in dataframe.keys().values) \
                and ("cloudTopRelativeChange" in dataframe.keys().values)\
                    and ("lwpEnd" in dataframe.keys().values):
                break
            
            lwpRelativeChange = []
            cloudTopRelativeChange = []
            lwpEndValue = []
            
            for emulInd, emul in enumerate(self.simulationCollection[trainingSet]):
                
                
                lwpStart = dataframe.loc[emul]["lwp"]
                print(lwpStart)
                lwpEnd = self.simulationCollection[trainingSet][emul].getTSDataset()["lwp_bar"].values[-1]*1000.
                print(lwpEnd)
                
                cloudTopStart = dataframe.loc[emul]["pblh_m"]
                cloudTopEnd = self.simulationCollection[trainingSet][emul].getTSDataset()["zc"].values[-1]
                
                
                lwpRelativeChange.append(lwpEnd / lwpStart)
                cloudTopRelativeChange.append(cloudTopEnd / cloudTopStart)
                lwpEndValue.append(lwpEnd)
                
            dataframe["lwpRelativeChange"] = lwpRelativeChange
            dataframe["cloudTopRelativeChange"] = cloudTopRelativeChange
            dataframe["lwpEndValue"] = lwpEndValue
            dataframe.to_csv( self.emulatorPostprosDataRootFolder / ( trainingSet + ".csv" )  )

    def getAnomalies(self):
        anomalyQuantile = 0.02
        self.anomalies ={}
        
        for trainingSet in self.trainingSetList:
            self.anomalies[trainingSet] = {}
            
            sVSpDataframe = self.simulatedVSPredictedCollection[trainingSet]
            simulDataFrame = self.simulationDataFrame[trainingSet]
            
            useQuantile = False
            if useQuantile:
                self.anomalyLimitTpot_inv = sVSpDataframe["tpot_inv"].quantile(anomalyQuantile)
                self.anomalyLimitCloudTopRelativeChange = simulDataFrame["cloudTopRelativeChange"].quantile(1-anomalyQuantile)
                self.anomalyLimitLWPRelativeChange = simulDataFrame["lwpRelativeChange"].quantile(1-anomalyQuantile)
                print("anomalyLimitTpot_inv", self.anomalyLimitTpot_inv)
                print("anomalyLimitCloudTopRelativeChange", self.anomalyLimitCloudTopRelativeChange)
                print("anomalyLimitLWPRelativeChange", self.anomalyLimitLWPRelativeChange)
            else:
                
                self.anomalyLimitTpot_inv = 2.5
                self.anomalyLimitCloudTopRelativeChange =  1.1
                self.anomalyLimitLWPRelativeChange = 1.4
            
            anomalyLimitQ_inv = sVSpDataframe["q_inv"].quantile(anomalyQuantile)
            
            
            
            
            
            print("anomalyLimitQ_inv",anomalyLimitQ_inv)
            
            
            print("tpot_pbl", sVSpDataframe["tpot_pbl"].quantile(0.05), sVSpDataframe["tpot_pbl"].quantile(0.95))
            
            self.anomalies[trainingSet]["tpot_inv_low_tail"] = ManuscriptFigures.getVariableAnomaly( sVSpDataframe, "tpot_inv", limit = self.anomalyLimitTpot_inv, highTail=False)
            self.anomalies[trainingSet]["q_inv_low_tail"] = ManuscriptFigures.getVariableAnomaly( sVSpDataframe, "q_inv", limit =  anomalyLimitQ_inv, highTail =False )
            
            self.anomalies[trainingSet]["cloudTopRelativeChange_high_tail"] =  [ int(caseID[3:])-1 for caseID in list(simulDataFrame["ID"].where(
                                            simulDataFrame["cloudTopRelativeChange"] > self.anomalyLimitCloudTopRelativeChange).dropna().values) ]
            
            self.anomalies[trainingSet]["lwpRelativeChange_high_tail"] =  [ int(caseID[3:])-1 for caseID in list(simulDataFrame["ID"].where(
                                            simulDataFrame["lwpRelativeChange"] > self.anomalyLimitLWPRelativeChange).dropna().values) ]
            
            self.anomalies[trainingSet]["tpot_high_tail"] =  ManuscriptFigures.getVariableAnomaly( sVSpDataframe, "tpot_pbl", limit =300, highTail = True)
            self.anomalies[trainingSet]["tpot_low_tail"] =  ManuscriptFigures.getVariableAnomaly( sVSpDataframe, "tpot_pbl", limit =273, highTail = False)
            
            
            
            self.anomalies[trainingSet]["pbl_low_tail"] =  ManuscriptFigures.getVariableAnomaly( sVSpDataframe, "pblh", percentile =0.05, highTail = False)
            self.anomalies[trainingSet]["pbl_high_tail"] = ManuscriptFigures.getVariableAnomaly( sVSpDataframe, "pblh", percentile =0.95, highTail = True)
            
            self.anomalies[trainingSet]["lwp_low_tail"] =  ManuscriptFigures.getVariableAnomaly( sVSpDataframe, "lwp", percentile =0.05, highTail = False)
            self.anomalies[trainingSet]["lwp_high_tail"] =  ManuscriptFigures.getVariableAnomaly( sVSpDataframe, "lwp", percentile =0.95, highTail = True)
            
            print(trainingSet[3])
            if trainingSet[3] == "3":
                self.anomalies[trainingSet]["aero_low_tail"] =  ManuscriptFigures.getVariableAnomaly( sVSpDataframe, "cdnc", percentile =0.05, highTail = False)
                self.anomalies[trainingSet]["aero_high_tail"] =  ManuscriptFigures.getVariableAnomaly( sVSpDataframe, "cdnc", percentile =0.95, highTail = True)
            else:
                self.anomalies[trainingSet]["aero_low_tail"] =  ManuscriptFigures.getVariableAnomaly( sVSpDataframe, "rdry_AS_eff", percentile =0.05, highTail = False)
                self.anomalies[trainingSet]["aero_high_tail"] =  ManuscriptFigures.getVariableAnomaly( sVSpDataframe, "rdry_AS_eff", percentile =0.95, highTail = True)
            
            if trainingSet[4:] == "Day":
                print("Day cos_mu",sVSpDataframe["cos_mu"].quantile(0.05),  sVSpDataframe["cos_mu"].quantile(0.95))
                self.anomalies[trainingSet]["cos_mu_low_quart"] = ManuscriptFigures.getVariableAnomaly( sVSpDataframe, "cos_mu", limit = math.cos(5*math.pi/12), highTail = False)
                self.anomalies[trainingSet]["cos_mu_high_quart"] = ManuscriptFigures.getVariableAnomaly( sVSpDataframe, "cos_mu", limit = math.cos(math.pi/12), highTail = True)
                
            else:
                self.anomalies[trainingSet]["cos_mu_low_quart"] = []
                self.anomalies[trainingSet]["cos_mu_high_quart"] = []
                
            
            print(" ")
    def getVariableAnomaly(dataframe, variablename, indexName= "designCase", limit= None, percentile=None, highTail = True):
        if limit is None:
            limit = dataframe[variablename].quantile(percentile)
            
        if highTail:
            factor = 1.
        else:
            factor = -1.
            
        return list(map(int, dataframe[indexName].where( factor * dataframe[variablename] > factor * limit ).dropna().values))
            
        
                
        
    def getPairedColorList(numberOfElements):
        if numberOfElements %2 != 0:
            sys.exit("Not divisible by two")
            
        bright = seaborn.color_palette("hls", int(numberOfElements/2))
        dark = seaborn.hls_palette(int(numberOfElements/2), l=.3, s=.8)
        
        paired = []
        for i in range(int(numberOfElements/2)):
            paired.append(bright[i])
            paired.append(dark[i])
                    
                       
        return paired
        
    def getColorsForLabelsPaired(labels):
        labels = list(labels)
        uniqueLabels = []
        
        for label in labels:
            if label not in uniqueLabels:
                uniqueLabels.append(label)
        
        uniqueColors =  ManuscriptFigures.getPairedColorList( len(uniqueLabels) )
        labelColors = dict(zip(uniqueLabels, uniqueColors))
        
        colorList = []
        for label in labels:
            colorList.append(labelColors[label])
        
        return colorList, labelColors
    
    def getColorsForLabels(labels):
        labels = list(labels)
        uniqueLabels = []
        
        for label in labels:
            if label not in uniqueLabels:
                uniqueLabels.append(label)
        
        uniqueColors =  Colorful.getIndyColorList( len(uniqueLabels) )
        labelColors = dict(zip(uniqueLabels, uniqueColors))

        colorList = []
        for label in labels:
            colorList.append(labelColors[label])
        
        return colorList, labelColors
        
    def figurePieSensitivyData(self):
        
        
        fig = Figure(self.figurefolder,"figureSensitivityPie", ncols = 2, nrows = 3, left = 0.01, right=0.99, hspace = 0.01, bottom=0.01 )
        
        
        allLabels = []
        for ind,trainingSet in enumerate(self.trainingSetList):
            maineffectLabel = [ label + " ME"  for label in  self.sensitivityDataCollection[trainingSet]["designVariableNames"].values ]
            interactionLabel = [ label + " IA" for label in  self.sensitivityDataCollection[trainingSet]["designVariableNames"].values ]
            stackedLabels = numpy.vstack((maineffectLabel,interactionLabel)).T.reshape(self.sensitivityDataCollection[trainingSet].shape[0]*2,)
            
            allLabels = numpy.concatenate((stackedLabels, allLabels))
            
        colorList, labelColors = ManuscriptFigures.getColorsForLabelsPaired(allLabels)
        
        for ind,trainingSet in enumerate(self.trainingSetList):
            print(ind,trainingSet)
            ax = fig.getAxes(ind)
            
            maineffect = self.sensitivityDataCollection[trainingSet]["MainEffect"]
            interaction = self.sensitivityDataCollection[trainingSet]["Interaction"]
            stackedData = numpy.vstack((maineffect,interaction)).T.reshape(self.sensitivityDataCollection[trainingSet].shape[0]*2,)
            
            maineffectLabel = [ label + " ME"  for label in  self.sensitivityDataCollection[trainingSet]["designVariableNames"].values ]
            interactionLabel = [ label + " IA" for label in  self.sensitivityDataCollection[trainingSet]["designVariableNames"].values ]
            stackedLabels = numpy.vstack((maineffectLabel,interactionLabel)).T.reshape(self.sensitivityDataCollection[trainingSet].shape[0]*2,)
            
            colors = []
            for lab in stackedLabels:
                colors.append(labelColors[lab])
            
            ax.pie( stackedData, colors=colors )
            PlotTweak.setAnnotation(ax, self.annotationCollection[trainingSet], xPosition=-1.1, yPosition = 1.1)
            
            
        
        fig.getAxes(4).axis("off")
        fig.getAxes(4).legend( handles=PlotTweak.getPatches(labelColors),
                              title = "Sensitivity",
                      loc=(0.25,0),
                      ncol = 3,
                      fontsize = 8)
        fig.getAxes(5).axis("off")  
        
        fig.save()
        
    def figureBarSensitivyData(self):
        
        
        fig = Figure(self.figurefolder,"figureSensitivityBar", figsize=(12/2.54,6),  ncols = 2, nrows = 2, hspace=0.5, bottom=0.32)
        
        grey = Colorful.getDistinctColorList("grey")
        allLabels = []
        for ind,trainingSet in enumerate(self.trainingSetList):
            allLabels = numpy.concatenate((self.sensitivityDataCollection[trainingSet]["designVariableNames"].values, allLabels))
        
        colorList, labelColors = ManuscriptFigures.getColorsForLabels(allLabels)
        
        legendLabelColors = {}
        for ind,trainingSet in enumerate(self.trainingSetList):
            ax = fig.getAxes(ind)
            
            dataframe = self.sensitivityDataCollection[trainingSet].sort_values(by=["MainEffect"], ascending=False)
            dataframe["mathLabel"] = dataframe.apply(lambda row: PlotTweak.getMathLabel(row.designVariableNames), axis=1)

            nroVariables = dataframe.shape[0]
            margin_bottom = numpy.zeros(nroVariables)
            
            oneColorList = []
            
            for variable in dataframe["designVariableNames"]:
                indColor = labelColors[variable]
                oneColorList.append(indColor)
                
                mathlabel = dataframe.set_index("designVariableNames").loc[variable]["mathLabel"]
                
                if mathlabel not in legendLabelColors:
                    legendLabelColors[mathlabel] = indColor
            
            for k,key in enumerate(["Interaction","MainEffect"]):
                if key == "MainEffect":
                    color = oneColorList
                else:
                    color = grey
                dataframe.plot(ax=ax, kind="bar",color=color, stacked=True, x="mathLabel", y = key, legend = False)
                
                margin_bottom += dataframe[key].values
            
            yTicks = numpy.arange(0, 0.51, 0.1)
            yTickLabels = [f"{t:.1f}" for t in yTicks]
            ax.set_yticks(yTicks)
            ax.set_yticklabels(yTickLabels)
            ax.set_ylim([0, 0.5])
            PlotTweak.setAnnotation(ax, self.annotationCollection[trainingSet], xPosition=ax.get_xlim()[1]*0.20, yPosition = ax.get_ylim()[1]*0.90)
            PlotTweak.setXaxisLabel(ax,"")
        
        labelColors["Interaction"] = grey
        
        
        
        fig.getAxes(0).legend(handles=PlotTweak.getPatches(legendLabelColors),
                                title = "Global variance -based sensitivity for " + PlotTweak.getLatexLabel("w_{pos}", False),
                      loc=(-0.2,-2.6),
                      ncol = 4,
                      fontsize = 8)
        fig.save()
    
    def figureLeaveOneOut(self):
        
        
        fig = Figure(self.figurefolder,"figureLeaveOneOut", figsize = [4.724409448818897, 4],  ncols = 2, nrows = 2, bottom = 0.12, hspace = 0.08, wspace=0.04, top=0.88)
        end = 0.8
        ticks = numpy.arange(0, end + .01, 0.1)
        tickLabels = [f"{t:.1f}" for t in ticks]
        
        showList = Data.cycleBoolean(len(ticks))
        
        showList[-1] = False
        
        for ind,trainingSet in enumerate(self.trainingSetList):
            ax = fig.getAxes(ind)
            
            dataframe = self.simulatedVSPredictedCollection[trainingSet]
            
            simulated = dataframe["wpos_Simulated"]
            emulated  = dataframe["wpos_Emulated"]
            
            
            
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(simulated, emulated)
            
            rSquared = numpy.power(r_value, 2)

            # #########################
            # bootStrapMean, bootStrapStd = self.getBootstrapMeanAverage(dataframe, "wpos_Simulated", "wpos_Emulated")
            
            print(" ")
            print("Leave-one-out")
            print(trainingSet, "mean absolute error", numpy.mean(numpy.abs(simulated-emulated)))
            print( "bootstrap", "mean", bootStrapMean, "std", bootStrapStd )
            ###################
            
            dataframe.plot.scatter(ax = ax, x="wpos_Simulated", y="wpos_Emulated",alpha=0.3)
            
            dataframeAnomalies = dataframe.loc[dataframe["designCase"].isin(self.anomalies[trainingSet]["cloudTopRelativeChange_high_tail"])]
            dataframeAnomalies.plot.scatter(ax = ax, x="wpos_Simulated", y="wpos_Emulated", color = self.cloudTopColor, marker = "x", linewidth = 1)
            
            dataframeAnomalies = dataframe.loc[dataframe["designCase"].isin(self.anomalies[trainingSet]["lwpRelativeChange_high_tail"])]
            dataframeAnomalies.plot.scatter(ax = ax, x="wpos_Simulated", y="wpos_Emulated", color = "blue", marker = "_", linewidth = 1)
            
            dataframeAnomalies = dataframe.loc[dataframe["designCase"].isin(self.anomalies[trainingSet]["tpot_inv_low_tail"])]
            dataframeAnomalies.plot.scatter(ax = ax, x="wpos_Simulated", y="wpos_Emulated", color = self.tempColor, marker = "|", linewidth = 1)
            
            # dataframeAnomalies = dataframe.loc[dataframe["designCase"].isin(self.anomalies[trainingSet]["aero_low_tail"])]
            # dataframeAnomalies.plot.scatter(ax = ax, x="wpos_Simulated", y="wpos_Emulated", edgecolors = self.tempColor, marker = "o", linewidth = 0.5, color='none')
            
            # dataframeAnomalies = dataframe.loc[dataframe["designCase"].isin(self.anomalies[trainingSet]["aero_high_tail"])]
            # dataframeAnomalies.plot.scatter(ax = ax, x="wpos_Simulated", y="wpos_Emulated", edgecolors = "k", marker = "o", linewidth = 0.5, color='none')
            
            # dataframeAnomalies = dataframe.loc[dataframe["designCase"].isin(self.anomalies[trainingSet]["cos_mu_low_quart"])]
            # dataframeAnomalies.plot.scatter(ax = ax, x="wpos_Simulated", y="wpos_Emulated", edgecolors = self.tempColor, marker = "o", linewidth = 0.5, color='none')
            
            # dataframeAnomalies = dataframe.loc[dataframe["designCase"].isin(self.anomalies[trainingSet]["cos_mu_high_quart"])]
            # dataframeAnomalies.plot.scatter(ax = ax, x="wpos_Simulated", y="wpos_Emulated", edgecolors = "k", marker = "o", linewidth = 0.5, color='none')
            
            coef = [slope, intercept]
            poly1d_fn = numpy.poly1d(coef)
            ax.plot(simulated.values, poly1d_fn(simulated.values), color = "k")
            
            ax.set_ylim([0, end])
            
            ax.set_xlim([0, end])
            
            
            PlotTweak.setAnnotation(ax, self.annotationCollection[trainingSet],
                                    xPosition=ax.get_xlim()[1]*0.05, yPosition = ax.get_ylim()[1]*0.90)
            
            PlotTweak.setAnnotation(ax, PlotTweak.getLatexLabel(f"R^2={rSquared:.2f}",""), xPosition=0.5, yPosition=0.1, bbox_props = None)
            
            PlotTweak.setXaxisLabel(ax,"")
            PlotTweak.setYaxisLabel(ax,"")
            
            if ind == 0:
                legendLabelColors = []
                
                legendLabelColors.append(matplotlib.lines.Line2D([], [], color=self.cloudTopColor, marker='x', markersize = 12, linestyle='None',
                          label='Cloud top rel. change >' + str(self.anomalyLimitCloudTopRelativeChange)))
                legendLabelColors.append(matplotlib.lines.Line2D([], [], color=self.lwpColor, marker='_', markersize = 12, linestyle="None",
                          label='LWP rel. change >' + str(self.anomalyLimitLWPRelativeChange)))
                legendLabelColors.append(matplotlib.lines.Line2D([], [], color=self.tempColor, marker='|', markersize = 12, linestyle='None',
                          label=r"$\Delta {\theta_{L}} < $" + str(self.anomalyLimitTpot_inv)))
                
                
                artist = ax.legend( handles=legendLabelColors, loc=(0.17, 1.05), frameon = True, framealpha = 1.0, ncol = 2 )
        
                ax.add_artist(artist)
            
            if ind in [2,3]:
                
                
                ax.set_xticks(ticks)
                ax.set_xticklabels(tickLabels)
                PlotTweak.hideLabels(ax.xaxis, showList)
            else:
                PlotTweak.hideXTickLabels(ax)
            
            if ind in [0,2]:
                ax.set_yticks(ticks)
                ax.set_yticklabels(tickLabels)
                PlotTweak.hideLabels(ax.yaxis, showList)
            else:
                
                PlotTweak.hideYTickLabels(ax)
            
            if ind == 2:
                ax.text(0.5,-0.2, PlotTweak.getUnitLabel("Simulated\ w_{pos}", "m\ s^{-1}"), size=8)
            if ind == 0:
                ax.text(-0.2,-0.5, PlotTweak.getUnitLabel("Emulated\ w_{pos}", "m\ s^{-1}"), size=8 , rotation =90)
                
        fig.save()
    
    def figureUpdraftLinearFit(self):
        
        
        fig = Figure(self.figurefolder,"figureLinearFit", figsize = [4.724409448818897, 5], ncols = 2, nrows = 2, bottom = 0.11, hspace = 0.08, wspace=0.12, top=0.86)
        xstart = -140
        xend = 50
        
        ystart = 0.0
        yend = 1.0
        yticks = numpy.arange(0, yend + .01, 0.1)
        tickLabels = [f"{t:.1f}" for t in yticks]
        
        yShowList = Data.cycleBoolean(len(yticks))
        
        color_obs = Colorful.getDistinctColorList("grey")
        
        for ind,trainingSet in enumerate(self.trainingSetList):
            ax = fig.getAxes(ind)
            
            dataframe = self.responseDataCollection[trainingSet]
            dataframe = dataframe[ dataframe["wpos"] != -999. ]
            dataframe = dataframe[ dataframe["prcp"] < 1e-6 ]
            
            
            radiativeWarming  = dataframe["drflx"].values
            updraft =  dataframe["wpos"].values
            
            slope_obs = -0.44
            intercept_obs = 22.30
            error_obs = 13./100.
            poly1d_Observation = numpy.poly1d(numpy.asarray([slope_obs,intercept_obs ])/100.) #?0.44 ×CTRC+
            ax.plot(radiativeWarming, poly1d_Observation(radiativeWarming), color = color_obs)
            ax.fill_between(sorted(radiativeWarming),
                            poly1d_Observation(sorted(radiativeWarming)) - error_obs*numpy.ones(numpy.shape(radiativeWarming)), poly1d_Observation(sorted(radiativeWarming)) + error_obs*numpy.ones(numpy.shape(radiativeWarming)),
                            alpha=0.2)
            
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(radiativeWarming, updraft)
            coef = [slope, intercept]
            rSquared = numpy.power(r_value, 2)
            
            dataColor = Colorful.getDistinctColorList("red")
            fitColor = "k"
            dataframe.plot.scatter(ax = ax, x="drflx", y="wpos",alpha=0.3, color = dataColor)
            
            tempAnomalies = [ k+1 for k in self.anomalies[trainingSet]["tpot_inv_low_tail"] ]
            cloudTopAnomalies = [ k+1 for k in self.anomalies[trainingSet]["cloudTopRelativeChange_high_tail"] ]
            lwpAnomalies = [ k+1 for k in self.anomalies[trainingSet]["lwpRelativeChange_high_tail"] ]
            
            tpotHighAnomalies = [ k+1 for k in self.anomalies[trainingSet]["tpot_high_tail"] ]
            tpotLowAnomalies = [ k+1 for k in self.anomalies[trainingSet]["tpot_low_tail"] ]
            
            cosMuHighAnomalies = [ k+1 for k in self.anomalies[trainingSet]["cos_mu_high_quart"] ]
            cosMuLowAnomalies = [ k+1 for k in self.anomalies[trainingSet]["cos_mu_low_quart"] ]
            
            
            dataframeAnomalies = dataframe.loc[dataframe["i"].isin(cloudTopAnomalies)]
            dataframeAnomalies.plot.scatter(ax = ax, x="drflx", y="wpos", color = self.cloudTopColor, marker = "x", linewidth = 1)
            
            dataframeAnomalies = dataframe.loc[dataframe["i"].isin(tempAnomalies)]
            dataframeAnomalies.plot.scatter(ax = ax, x="drflx", y="wpos", color = self.tempColor, marker = "|", linewidth = 1)
            
            dataframeAnomalies = dataframe.loc[dataframe["i"].isin(lwpAnomalies)]
            dataframeAnomalies.plot.scatter(ax = ax, x="drflx", y="wpos", color = self.lwpColor, marker = "_", linewidth = 1)
            
            # dataframeAnomalies = dataframe.loc[dataframe["i"].isin(tpotHighAnomalies)]
            # dataframeAnomalies.plot.scatter(ax = ax, x="drflx", y="wpos", edgecolors = "k", marker = "o", linewidth = 0.5, color='none')
            
            # dataframeAnomalies = dataframe.loc[dataframe["i"].isin(tpotLowAnomalies)]
            # dataframeAnomalies.plot.scatter(ax = ax, x="drflx", y="wpos", edgecolors = self.tempColor, marker = "o", linewidth = 0.5, color='none')
            
            # dataframeAnomalies = dataframe.loc[dataframe["i"].isin(cosMuHighAnomalies)]
            # dataframeAnomalies.plot.scatter(ax = ax, x="drflx", y="wpos", edgecolors = "k", marker = "o", linewidth = 0.5, color='none')
            
            # dataframeAnomalies = dataframe.loc[dataframe["i"].isin(cosMuLowAnomalies)]
            # dataframeAnomalies.plot.scatter(ax = ax, x="drflx", y="wpos", edgecolors = self.tempColor, marker = "o", linewidth = 0.5, color='none')
            
            poly1d_fn = numpy.poly1d(coef)
            
            linearFit = []
            for radWarmingValue in list(self.responseDataCollection[trainingSet]["drflx"]):
                linearFit.append(poly1d_fn(radWarmingValue))
            
            self.responseDataCollection[trainingSet]["linearFit"] =linearFit #dataframe.apply(lambda row: poly1d_fn(row.drflx), axis = 1)
            ax.plot(radiativeWarming, poly1d_fn(radiativeWarming), color = fitColor)
            
            self.responseDataCollection[trainingSet].to_csv("/home/aholaj/Data/EmulatorManuscriptData/Datasets/" + trainingSet + "_fit.csv")
            ax.set_xlim([xstart, xend])
            ax.set_ylim([ystart, yend])
            
            
            PlotTweak.setAnnotation(ax, self.annotationCollection[trainingSet],
                                    xPosition=PlotTweak.getXPosition(ax, 0.02), yPosition = PlotTweak.getYPosition(ax, 0.94))

            PlotTweak.setXaxisLabel(ax,"")
            PlotTweak.setYaxisLabel(ax,"")
            
            
            xticks = PlotTweak.setXticks(ax, start= xstart, end = xend, interval = 10, integer=True)
        
            xShownLabelsBoolean = PlotTweak.setXLabels(ax, xticks, start= xstart, end = xend, interval = 40)
            xShownLabelsBoolean = Data.cycleBoolean(len(xShownLabelsBoolean))
            PlotTweak.setXTickSizes(ax, xShownLabelsBoolean)
            
            
            if ind == 0:
                collectionOfLabelsColors = {"Simulated data": dataColor, "Fit" : "k", "Observations" : color_obs}
                legendLabelColors = PlotTweak.getPatches(collectionOfLabelsColors)
                
                legendLabelColors.append(matplotlib.lines.Line2D([], [], color=self.cloudTopColor, marker='x', markersize = 12, linestyle='None',
                          label='Cloud top rel. change >' + str(self.anomalyLimitCloudTopRelativeChange)))
                legendLabelColors.append(matplotlib.lines.Line2D([], [], color=self.tempColor, marker='|', markersize = 12, linestyle='None',
                          label=r"$\Delta {\theta_{L}} < $" + str(self.anomalyLimitTpot_inv)))
                legendLabelColors.append(matplotlib.lines.Line2D([], [], color=self.lwpColor, marker='_', markersize = 12, linestyle="None",
                          label='LWP rel. change >' + str(self.anomalyLimitLWPRelativeChange)))
                
                
                legendLabelColors = list(numpy.asarray(legendLabelColors).reshape(3,2).T.flatten())
                artist = ax.legend( handles=legendLabelColors, loc=(0.17, 1.05), frameon = True, framealpha = 1.0, ncol = 2 )
        
                ax.add_artist(artist)
            
            ax.text(-30, 0.75,
                PlotTweak.getLatexLabel("y=a + b * x") + "\n" + \
                                         PlotTweak.getLatexLabel(f"a={intercept:.4f}") + "\n" + \
                                             PlotTweak.getLatexLabel(f"b={slope:.6f}") + "\n" + \
                                           PlotTweak.getLatexLabel(f"R^2={rSquared:.2f}"), fontsize = 6)
            
            ax.set_yticks(yticks)
            ax.set_yticklabels(tickLabels)
            PlotTweak.setYTickSizes(ax, yShowList)
            PlotTweak.hideLabels(ax.yaxis, yShowList)
            
            
            
            if ind in [1,3]:
                PlotTweak.hideYTickLabels(ax)
            
            if ind in [0,1]:
                PlotTweak.hideXTickLabels(ax)
            if ind == 0:
                ax.text(PlotTweak.getXPosition(ax, -0.27), PlotTweak.getYPosition(ax, -0.5),
                        PlotTweak.getUnitLabel("Simulated\ w_{pos}", "m\ s^{-1}"), size=8 , rotation =90)
            if ind == 2:
                ax.text(0.3,-0.25, PlotTweak.getUnitLabel("Cloud\ rad.\ warming", "W\ m^{-2}"), size=8)
                
        fig.save()
        
    def figureUpdraftLinearFitVSEMul(self):
        fig = Figure(self.figurefolder,"figureLinearFitComparison", figsize = [4.724409448818897, 4.5], 
                     ncols = 2, nrows = 2, bottom = 0.11, hspace = 0.08, wspace=0.12, top=0.86)
        # xticks = numpy.arange(0, xend + 1, 10)
        
        start = 0.0
        end = 1.0
        ticks = numpy.arange(0, end + .01, 0.1)
        tickLabels = [f"{t:.1f}" for t in ticks]
        
        showList = Data.cycleBoolean(len(ticks))
        
        showList[-1] = False
        
        for ind,trainingSet in enumerate(self.trainingSetList):
            ax = fig.getAxes(ind)
            
            dataframeFit = self.responseDataCollection[trainingSet]
            dataframeFit = dataframeFit[ dataframeFit["wpos"] != -999. ]
            simulatedFit = dataframeFit["wpos"]
            fittedFit = dataframeFit["linearFit"]
            
            slopeFit, interceptFit, r_valueFit, p_valueFit, std_errFit = scipy.stats.linregress(simulatedFit, fittedFit)
            
            rSquaredFit = numpy.power(r_valueFit,2)
            
            ###########################
            # bootStrapMean, bootStrapStd = self.getBootstrapMeanAverage(dataframeFit, "wpos", "linearFit")
          
            
            print(" ")
            print("Linear-fit vs emulator")
            print(trainingSet, "mean absolute error", numpy.mean(numpy.abs(simulatedFit-fittedFit)))
            print( "bootstrap", "mean", bootStrapMean, "std", bootStrapStd )
            ######################
            
            coefFit = [slopeFit, interceptFit]
            
            
            fitColor = "k"
            dataColor = Colorful.getDistinctColorList("red")
            
            tempAnomalies = [ k+1 for k in self.anomalies[trainingSet]["tpot_inv_low_tail"] ]
            cloudTopAnomalies = [ k+1 for k in self.anomalies[trainingSet]["cloudTopRelativeChange_high_tail"] ]
            lwpAnomalies = [ k+1 for k in self.anomalies[trainingSet]["lwpRelativeChange_high_tail"] ]
            
            dataframeFit.plot.scatter(ax=ax, x="wpos", y="linearFit", alpha = 0.3, color=dataColor)
            
            dataframeAnomalies = dataframeFit.loc[dataframeFit["i"].isin(cloudTopAnomalies)]
            dataframeAnomalies.plot.scatter(ax = ax, x="wpos", y="linearFit", color = self.cloudTopColor, marker = "x", linewidth = 1)
            
            dataframeAnomalies = dataframeFit.loc[dataframeFit["i"].isin(tempAnomalies)]
            dataframeAnomalies.plot.scatter(ax = ax, x="wpos", y="linearFit", color = self.tempColor, marker = "|", linewidth = 1)
            
            dataframeAnomalies = dataframeFit.loc[dataframeFit["i"].isin(lwpAnomalies)]
            dataframeAnomalies.plot.scatter(ax = ax, x="wpos", y="linearFit", color = self.lwpColor, marker = "_", linewidth = 1)
        
            poly1d_fn = numpy.poly1d(coefFit)
            
            ax.plot(simulatedFit.values, poly1d_fn(simulatedFit.values), color = fitColor)
            
            
            ax.set_xlim([start, end])
            ax.set_ylim([start, end])
            
            
            PlotTweak.setAnnotation(ax, self.annotationCollection[trainingSet],
                                    xPosition=ax.get_xlim()[1]*0.05, yPosition = ax.get_ylim()[1]*0.90)
            
            PlotTweak.setAnnotation(ax, PlotTweak.getLatexLabel(f"R^2={rSquaredFit:.2f}",""), xPosition=0.5, yPosition=0.1, bbox_props = None)
            
            PlotTweak.setAnnotation(ax, self.annotationCollection[trainingSet],
                                    xPosition=PlotTweak.getXPosition(ax, 0.05), yPosition = PlotTweak.getYPosition(ax, 0.9))
            
            PlotTweak.setXaxisLabel(ax,"")
            PlotTweak.setYaxisLabel(ax,"")
            
            
            PlotTweak.setXTickSizes(ax, showList)
            
            
            ax.set_xticks(ticks)
            ax.set_xticklabels(tickLabels)
            PlotTweak.hideLabels(ax.xaxis, showList)
            
            if ind == 0:
                collectionOfLabelsColors = {"Simulated data": dataColor, "Fit" : "k"}
                legendLabelColors = PlotTweak.getPatches(collectionOfLabelsColors)
                
                legendLabelColors.append(matplotlib.lines.Line2D([], [], color=self.cloudTopColor, marker='x', markersize = 12, linestyle='None',
                          label='Cloud top rel. change >' + str(self.anomalyLimitCloudTopRelativeChange)))
                legendLabelColors.append(matplotlib.lines.Line2D([], [], color=self.tempColor, marker='|', markersize = 12, linestyle='None',
                          label=r"$\Delta {\theta_{L}} < $"+ str(self.anomalyLimitTpot_inv)))
                legendLabelColors.append(matplotlib.lines.Line2D([], [], color=self.lwpColor, marker='_', markersize = 12, linestyle="None",
                          label='LWP rel. change >' + str(self.anomalyLimitLWPRelativeChange)))
                legendLabelColors.append(None)
                
                
                legendLabelColors = list(numpy.asarray(legendLabelColors).reshape(3,2).T.flatten())
                # print(legendLabelColors)
                legendLabelColors.pop(-1)
                # print(legendLabelColors)
                artist = ax.legend( handles=legendLabelColors, loc=(0.17, 1.05), frameon = True, framealpha = 1.0, ncol = 2 )
        
                ax.add_artist(artist)
                
            ax.set_yticks(ticks)
            ax.set_yticklabels(tickLabels)
            PlotTweak.setYTickSizes(ax, showList)
            PlotTweak.hideLabels(ax.yaxis, showList)
            
            
            
            if ind in [1,3]:
                PlotTweak.hideYTickLabels(ax)
            if ind in [0,1]:
                PlotTweak.hideXTickLabels(ax)
            
            if ind == 2:
                ax.text(0.5,-.25, PlotTweak.getUnitLabel("Updraft\ from\ emulator", "m\ s^{-1}"), size=8)
            if ind == 0:
                ax.text(PlotTweak.getXPosition(ax, -0.27), PlotTweak.getYPosition(ax, -0.4), PlotTweak.getUnitLabel("Updraft\ from\ linear\ fit", "m\ s^{-1}"), size=8 , rotation =90)
                
        fig.save()
        
    def figureErrorDistribution(self):
        fig = Figure(self.figurefolder,"figureErrorDistribution",  ncols = 2, nrows = 2,
                     bottom = 0.15, hspace = 0.08, wspace=0.04, top=0.90)
        mini = None
        maxi = None
        ymaxi = None
        xticks = numpy.arange(-0.4, 0.21, 0.1)
        xtickLabels = [f"{t:.1f}" for t in xticks]
        xshowList = Data.cycleBoolean(len(xticks))
        xshowList[-1] = False
        
        yticks = numpy.arange(0, 141, 10)
        ytickLabels = [f"{t:d}" for t in yticks]
        yshowList = Data.getMaskedList(yticks, numpy.arange(0,141,50))
        # yshowList[-1] = False
        emulColor = Colorful.getDistinctColorList("blue")
        linFitColor = Colorful.getDistinctColorList("red")
        for ind,trainingSet in enumerate(self.trainingSetList):
            ax = fig.getAxes(ind)
            
            dataframe = self.simulatedVSPredictedCollection[trainingSet]
            
            
            dataframe["absErrorEmul"] = dataframe.apply(lambda row: row.wpos_Simulated - row.wpos_Emulated, axis = 1)
            
            emulRMSE = sqrt(mean_squared_error(dataframe["wpos_Simulated"], dataframe["wpos_Emulated"]))
            
            # print(trainingSet, emulRMSE)
            
            dataframe["absErrorEmul"].plot.hist(ax=ax, bins = 20, color = emulColor,  style='--', alpha = 0.5 )
            
            if mini is None:
                mini = dataframe["absErrorEmul"].min()
            else:
                mini = min(mini, dataframe["absErrorEmul"].min())
            
            if maxi is None:
                maxi = dataframe["absErrorEmul"].max()
            else:
                maxi = max(maxi,  dataframe["absErrorEmul"].min())
                
            
            dataframe2 = self.responseDataCollection[trainingSet]
            
            
            dataframe2["absErrorLinearFit"] = dataframe2.apply(lambda row: row.wpos - row.linearFit, axis = 1)
            
            dataframe2Filtered = dataframe2[ dataframe2["wpos"] != -999. ]
            
            linfitRMSE = sqrt(mean_squared_error(dataframe2Filtered["wpos"], dataframe2Filtered["linearFit"]))
            
            dataframe2Filtered["absErrorLinearFit"].plot.hist(ax=ax, bins = 20, color = linFitColor, style='--', alpha = 0.5 )
            
            mini = min(mini, dataframe2Filtered["absErrorLinearFit"].min())
            maxi = max(maxi, dataframe2Filtered["absErrorLinearFit"].max())
            
            if ymaxi is None:
                ymaxi = ax.get_ylim()[1]
            else:
                ymaxi = max(ymaxi, ax.get_ylim()[1])                
            
            stringi = f"RMS errors:\nEmulated: {emulRMSE:.4f}"
            stringi = stringi + f"\nLinear Fit: {linfitRMSE:.4f}"
            PlotTweak.setAnnotation(ax, stringi,
                                    xPosition=-0.39, yPosition=60, bbox_props = None)
            
            ax.set_xlim([xticks[0], xticks[-1]])
            ax.set_ylim([0, yticks[-1]])
            PlotTweak.setXaxisLabel(ax,"")
            PlotTweak.setYaxisLabel(ax,"")
            
            PlotTweak.setAnnotation(ax, self.annotationCollection[trainingSet],
                                    xPosition=(ax.get_xlim()[1]-ax.get_xlim()[0])*0.02 + ax.get_xlim()[0], yPosition = ax.get_ylim()[1]*0.89)
            
            if ind in [2,3]:
                ax.set_xticks(xticks)
                ax.set_xticklabels(xtickLabels)
                PlotTweak.hideLabels(ax.xaxis, xshowList)
                PlotTweak.setXTickSizes(ax, xshowList)
            else:
                PlotTweak.hideXTickLabels(ax)
            
            if ind in [0,2]:
                ax.set_yticks(yticks)
                ax.set_yticklabels(ytickLabels)
                PlotTweak.hideLabels(ax.yaxis, yshowList)
                PlotTweak.setYTickSizes(ax, yshowList)
            else:
                
                PlotTweak.hideYTickLabels(ax)
            
            if ind == 2:
                ax.text(-0.3,-50, PlotTweak.getUnitLabel("Error,\ w-w_{LES},\ for\ predicting\ updraft\ velocity", "m\ s^{-1}"), size=8)
            if ind == 0:
                ax.text(-0.55,-80, PlotTweak.getLatexLabel("Number\ of\ points"), size=8 , rotation =90)
                PlotTweak.setArtist(ax, {"Updraft from emulator": emulColor, "Updraft from linear fit" : linFitColor}, loc = (0.07, 1.05), ncol = 2)
            
            #dataframe["emulRMSE"] = dataframe.apply(lambda col: sqrt(mean_squared_error(dataframe["wpos_Simulated"], dataframe["wpos_Emulated"])), axis = 0 )
        
        fig.save()    
        
    def plot4Sets(self, trainingSetList, simulationCollection, annotationCollection, simulationDataFrames,
                  figurefolder, figurename,
                  ncVariable, designVariable,
                  conversionNC = 1.0, conversionDesign = 1.0,
                  xmax = 1000, ymax = 1000,
                  xAxisLabel  = None, xAxisUnit = None,
                  yAxisLabel = None, yAxisUnit = None, keisseja = 10000,
                  yPositionCorrection = 100, outlierParameter = 0.2):
        
        relativeChangeDict  = Data.emptyDictionaryWithKeys(trainingSetList)
        # create figure object
        fig = Figure(figurefolder,figurename, ncols=2, nrows=2, sharex=True, sharey = True)
        # plot timeseries with unit conversion
        maks = 0
        mini = 0
        for ind, case in enumerate(trainingSetList):
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
            
        for ind, case in enumerate(trainingSetList):
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
    
    def getLWPFigure(self):
        lwpFig, lwpChangeParameters = plot4Sets(trainingSetList, simulationCollection, annotationCollection, simulationDataFrames, figurefolder, "lwp",
              "lwp_bar", "lwp",
              conversionNC = 1000., conversionDesign = 1.0,
              xmax = 1000, ymax = 1000,
              xAxisLabel = "LWP", xAxisUnit = "g m^{-2}",
              yAxisLabel = "LWP", yAxisUnit = "g m^{-2}", keisseja = keisseja, outlierParameter=0.5)
        
        simulationDataFrames = mergeDataFrameWithParam( simulationDataFrames, lwpChangeParameters, "lwpRel")
        
        lwpFig.save()
        
    def getCloudTopFigure(self):
            cloudTopFig, cloudTopParameters = plot4Sets(trainingSetList, simulationCollection, annotationCollection, simulationDataFrames, figurefolder, "cloudtop",
                  "zc", "pblh_m",
                  xmax = 3600, ymax = 3600,
                  xAxisLabel = "Cloud\ top", xAxisUnit = "m",
                  yAxisLabel = "Cloud\ top", yAxisUnit = "m", keisseja = keisseja,
                  yPositionCorrection=300)
            simulationDataFrames = mergeDataFrameWithParam( simulationDataFrames, cloudTopParameters, "zcRel")
            cloudTopFig.save()    
            
    def getLWPOutliers(self):
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
        
        for ind, case in enumerate(trainingSetList):
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

def main():
    
    figObject = ManuscriptFigures(os.environ["EMULATORPOSTPROSDATAROOTFOLDER"], 
                                  os.environ["EMULATORFIGUREFOLDER"])
    
    figObject.readSensitivityData()
    figObject.readSimulatedVSPredictedData()
    figObject.readResponseData()
    figObject.getOutlierDataFromLESoutput()
    figObject.getAnomalies()
    figObject.fillUpDrflxValues()
    
    if True:
        figObject.figureBarSensitivyData()
    if True:
        figObject.figureUpdraftLinearFit()
    if True:
        figObject.figureErrorDistribution()
    if True:
        figObject.figureLeaveOneOut()
    if True:
        figObject.figureUpdraftLinearFitVSEMul()
    
        
if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("Script completed in " + str(round((end - start),0)) + " seconds")
