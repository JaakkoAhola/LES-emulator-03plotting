#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 15:46:01 2020

@author: Jaakko Ahola, Finnish Meteorological Institute
@licence: MIT licence Copyright

Plot figures for Updraft emulator manuscript
"""
print(__doc__)
import copy
import matplotlib
import matplotlib.ticker as mticker
import math
import numpy
import os
import pandas
import pathlib
import sys
from scipy import stats
import time
from datetime import datetime

sys.path.append(os.environ["LESMAINSCRIPTS"])
from Colorful import Colorful
from Data import Data
from Figure import Figure
from PlotTweak import PlotTweak

from sklearn.metrics import mean_squared_error
from math import sqrt
from math import ceil

sys.path.append("../LES-emulator-02postpros")
from EmulatorMetaData import EmulatorMetaData

class ManuscriptFigures(EmulatorMetaData):

    def __init__(self, locationsFile):
        
        super().__init__(locationsFile)
        
        self.figures = {}
        
        self.figureWidth = 12/2.54
    
        self.trainingSetList = ["LVL3Night",
                               "LVL3Day",
                               "LVL4Night",
                               "LVL4Day"]
        
        self.trainingSetColors = dict(zip(self.trainingSetList, Colorful.getDistinctColorList( ["blue", "cyan", "red", "orange"])))

        
        self.figureFolder.mkdir( parents=True, exist_ok = True )

        self.tableFolder.mkdir( parents=True, exist_ok = True )

        self.annotationValues = ["(a) SB Night",
            "(b) SB Day",
            "(c) SALSA Night",
            "(d) SALSA Day"]
        
        self.trainingSetSensibleNames = ["SB\ Night",
            "SB\ Day",
            "SALSA\ Night",
            "SALSA\ Day"]
        
        self.trainingSetSensibleDict = dict(zip(self.trainingSetList, self.trainingSetSensibleNames))
        
        self.allSetColors = dict(zip([PlotTweak.getLatexLabel(name) for name in ["Filtered\ ECHAM"] + self.trainingSetSensibleNames], [Colorful.getDistinctColorList("grey")] + list(self.trainingSetColors.values()) ))

        self.annotationCollection = dict(zip(self.trainingSetList, self.annotationValues))

        self.cloudTopColor = Colorful.getDistinctColorList("green")
        self.lwpColor = Colorful.getDistinctColorList("blue")
        self.tempColor = Colorful.getDistinctColorList("yellow")

        self.observationParameters = {}
        self.observationParameters["slope"] = -0.44/100.
        self.observationParameters["intercept"] = 22.30/100.
        self.observationParameters["error"] = 13./100.

        self.predictorShortNames = ["linearFit", "correctedLinearFit", "emulator"]
        
        self.predictorShortNameDict = dict(zip(self.predictionVariableList, self.predictorShortNames))
        
        self.predictorColors = dict(zip(self.predictorShortNames, Colorful.getDistinctColorList(["red", "green", "blue"])))
            
        
        self.predictorClearNames = dict(zip(self.predictorShortNames, ["LF", "LFRF", "GPE"]))

        self._initReadCompleteData()

        self._filterCompleteData()

        self._initReadStats()

        self._initReadLimits()
        
        self._initBootstraps()
        
        self._initUpdraftTicks()
        
    def _initUpdraftTicks(self):
        self.sbTicks = numpy.arange(0, 0.8 + 0.01, 0.1)
        self.salsaTicks = numpy.arange(0,0.8  + 0.01, 0.1)
        
        self.sbTicksLabels = self._intTicksLabels( self.sbTicks)
        self.salsaTicksLabels = self._intTicksLabels( self.salsaTicks)
        
        self.sbDensityTicks = numpy.arange(0., 0.30+0.01, 0.05)
        self.salsaDensityTicks = numpy.arange(0., 0.025+0.01, 0.005)
    
        self.sbDensityTicksLabels = [f"{t:.2f}" for t in self.sbDensityTicks]
        self.salsaDensityTicksLabels = [f"{t:.3f}" for t in self.salsaDensityTicks]
    def _intTicksLabels(self,ticks):
        return [f"{t:.2f}" for t in ticks]
    

    def _initReadCompleteData(self):
        self.completeDataFrame = {}
        for trainingSet in self.trainingSetList:
            try:
                self.completeDataFrame[trainingSet] = pandas.read_csv( self.postProsDataRootFolder / trainingSet / ( trainingSet + "_complete.csv" ), index_col = 0  )
            except FileNotFoundError:
                self.completeDataFrame[trainingSet] = None
                
    def _filterCompleteData(self):
        self.completeDataFrameFiltered = {}
        for trainingSet in self.trainingSetList:
            if self.completeDataFrame[trainingSet] is not None:
                dataframe = self.completeDataFrame[trainingSet]
                dataframe = dataframe.loc[ dataframe[self.filterIndex] ]
                self.completeDataFrameFiltered[trainingSet] = dataframe
            else:
                self.completeDataFrameFiltered[trainingSet] = None

    def initReadFeatureImportanceData(self):

        self._initReadFeatureImportanceData_phase01()
        self._initReadFeatureImportanceData_phase02()
        self._initReadFeatureImportanceData_phase03()
        self._initReadFeatureImportanceData_phase04()
        self._initReadFeatureImportanceData_phase05()

    def _initReadFeatureImportanceData_phase01(self):
        
        self.featureImportanceDataCollection = {}
        self.allLabels = []
        self.legendCols = 4
        for trainingSet in self.trainingSetList:
            self.featureImportanceDataCollection[trainingSet] = {}
            try:
                dataset = pandas.read_csv( self.postProsDataRootFolder / trainingSet / ( trainingSet + "_featureImportance.csv" ), index_col = 0 )
                
            except FileNotFoundError:
                self.featureImportanceDataCollection[trainingSet] = None
                continue
            for ind in dataset.index:
                series = dataset.loc[ind]
                mean = series.loc[[kk for kk in series.index if kk[-4:] == "Mean"]]
                std = series.loc[[kk for kk in series.index if kk[-3:] == "Std"]]
                ines = [kk[:-22] for kk in series.index if kk[-4:] == "Mean"]
                
                self.allLabels += ines
                
                subset = pandas.DataFrame(data =  {"Mean" : mean.values, "Std": std.values}, index = ines).dropna()
                try:
                    subset = subset.drop(index="wpos_linearFit")
                except:
                    pass
                
                subset["mathLabel"] = subset.apply(lambda row: PlotTweak.getMathLabel(row.name), axis = 1)

                subset = subset.sort_values("Mean", ascending=False)
                subset["points"] = range(len(subset))
                summa = subset["Mean"].sum()

                subset["relativeImportance"] = subset.apply(lambda row: max(row["Mean"], 0) / summa, axis = 1)
                
                self.featureImportanceDataCollection[trainingSet][ind] = subset
        
        self.uniqueLabels = list(set(self.allLabels))
        
    def _initReadFeatureImportanceData_phase02(self):
        self.labelPoints = dict(zip(self.uniqueLabels, [0]*len(self.uniqueLabels)))
        for key in self.uniqueLabels:
            for row,trainingSet in enumerate(list(self.featureImportanceDataCollection)):
                if self.featureImportanceDataCollection[trainingSet] is None:
                        continue
                for col, predictor in enumerate(list(self.featureImportanceDataCollection[trainingSet])):
                    dataframe = self.featureImportanceDataCollection[trainingSet][predictor]
                    
                    if key in dataframe.index:
                        self.labelPoints[key] += dataframe.loc[key]["points"]
                        
        self.labelPoints = pandas.Series(self.labelPoints).sort_values().to_dict()
    
    def _initReadFeatureImportanceData_phase03(self):
        self.labelRelative = dict(zip(self.uniqueLabels, [1]*len(self.uniqueLabels)))
        
        self.zeros = dict(zip(self.uniqueLabels, [0]*len(self.uniqueLabels)))

        self.meanArray = {}
        for key in self.uniqueLabels:
            self.meanArray[key] = []
        
        self.methodMeanArray = {}
        for predictor in self.featureImportanceDataCollection["LVL3Night"]:
            self.methodMeanArray[predictor] = {}
            for key in self.uniqueLabels:
                self.methodMeanArray[predictor][key] = []
        
        for key in self.uniqueLabels:
            for row,trainingSet in enumerate(list(self.featureImportanceDataCollection)):
                if self.featureImportanceDataCollection[trainingSet] is None:
                    continue
                for col, predictor in enumerate(list(self.featureImportanceDataCollection[trainingSet])):
                    
                    
                    dataframe = self.featureImportanceDataCollection[trainingSet][predictor]
                    if key in dataframe.index:
                        relative = dataframe.loc[key]["relativeImportance"]
                        self.meanArray[key].append( dataframe.loc[key]["relativeImportance"] )
                        self.methodMeanArray[predictor][key].append(dataframe.loc[key]["relativeImportance"])
                        if relative < Data.getEpsilon():
                            relative = 1
                            self.zeros[key] += 1
                        self.labelRelative[key] *= relative
        
    def _initReadFeatureImportanceData_phase04(self):
        
        for key in self.meanArray:
            self.meanArray[key] = numpy.asarray(self.meanArray[key]).mean()
        for predictor in self.methodMeanArray:
            for key in self.methodMeanArray[predictor]:
                self.methodMeanArray[predictor][key] = numpy.asarray(self.methodMeanArray[predictor][key]).mean()
        
        relativeCombined = pandas.DataFrame.from_dict(self.labelRelative, orient="index", columns = ["relativeCombined"])
        zeros = pandas.DataFrame.from_dict(self.zeros, orient="index", columns = ["zeros"])
        
        allMean = pandas.DataFrame.from_dict(self.meanArray, orient="index", columns = ["relativeMean"])
        
        self.methodMeans = {}        
        for predictor in self.methodMeanArray:
            if predictor.split("_")[-1] == "Emulated":
                self.methodMeans[self.predictorClearNames["emulator"]] = self.methodMeanArray[predictor]
            elif predictor.split("_")[-1] == "CorrectedLinearFit":
                self.methodMeans[self.predictorClearNames["correctedLinearFit"]] = self.methodMeanArray[predictor]
                
                
        lfrfMean = pandas.DataFrame.from_dict(self.methodMeans[self.predictorClearNames["correctedLinearFit"]], orient="index", columns = ["LFRF relative mean"])
        gpeMean = pandas.DataFrame.from_dict(self.methodMeans[self.predictorClearNames["emulator"]], orient="index", columns = ["GPE relative mean"])
        
        self.labelCategorised = pandas.concat((allMean, relativeCombined, zeros, lfrfMean, gpeMean), axis = 1)
        
        try:
            self.labelCategorised = self.labelCategorised.drop(index="wpos_linearFit")
        except:
            pass
        
        self.labelCategorised["mathLabel"] = self.labelCategorised.apply(lambda row: PlotTweak.getMathLabel(row.name), axis = 1)
        self.labelCategorised = self.labelCategorised.sort_values(by="relativeMean", ascending =False)
        
        
    def _initReadFeatureImportanceData_phase05(self):
        self.uniqueLabels = self.labelCategorised.index.values
        
        self._getColorsForLabels()
        
        for trainingSet in self.featureImportanceDataCollection:
            if self.featureImportanceDataCollection[trainingSet] is None:
                continue
            for predictor in self.featureImportanceDataCollection[trainingSet]:
                subset  = self.featureImportanceDataCollection[trainingSet][predictor]
                subset["color"] = subset.apply(lambda row: self.labelColors[row.name], axis = 1)
        

    def initReadFilteredSourceData(self):
        localPath = pathlib.Path("/home/aholaj/Data/ECLAIR")
        if localPath.is_dir():
            try:
                self.filteredSourceData = pandas.read_csv( localPath / "eclair_dataset_2001.csv", index_col = 0 )
                print("FilteredSourceData locally")
            except FileNotFoundError:
                self.filteredSourceData = None
        else:
            try:
                self.filteredSourceData = pandas.read_csv( self.postProsDataRootFolder / "eclair_dataset_2001.csv", index_col = 0 )
            except FileNotFoundError:
                self.filteredSourceData = None
    
    def _initReadStats(self):
        self.statsCollection = {}
        for trainingSet in self.trainingSetList:
            try:
                self.statsCollection[trainingSet] = pandas.read_csv( self.postProsDataRootFolder / trainingSet / ( trainingSet + "_stats.csv" ), index_col = 0  )
            except FileNotFoundError:
                self.statsCollection[trainingSet] = None
            
    def _initBootstraps(self):
        self.bootstrapCollection = {}
        for trainingSet in self.trainingSetList:
            try:
                self.bootstrapCollection[trainingSet] = pandas.read_csv( self.postProsDataRootFolder / trainingSet / ( trainingSet + "_bootstrap.csv" ), index_col = 0  )
            except FileNotFoundError:
                self.bootstrapCollection[trainingSet] = None
            
    def _initReadLimits(self):
        try:
            self.anomalyLimits = pandas.read_csv( self.postProsDataRootFolder / "anomalyLimits.csv", index_col = 0)
        except FileNotFoundError:
            self.anomalyLimits = None

    def _getColorsForLabels(self):
        self.uniqueLabels = Data.reorganiseArrayByColumnNumber(self.uniqueLabels, self.legendCols)
        
        self.uniqueColors = [ PlotTweak.getLabelColor( label ) for label in self.uniqueLabels ]
        
        
        
        
        self.labelColors = dict(zip(self.uniqueLabels, self.uniqueColors))
        
        self.uniqueMathLabels = [PlotTweak.getMathLabel(label) for label in self.uniqueLabels]
        
        self.mathLabelColors = dict(zip(self.uniqueMathLabels, self.uniqueColors))

    def finalise(self):

        for fig in self.figures.values():
            fig.save(file_extension = ".pdf")


    def figureBarFeatureImportanceData(self):
        nrows=2
        ncols=2
        self.figures["figureFeatureImportanceBar"] = Figure(self.figureFolder,"figureFeatureImportanceBar",
                                                            figsize=[self.figureWidth,6], 
                                                            ncols = ncols, nrows = nrows,
                                                            hspace=0.3, bottom=0.20, wspace = 0.05, top = 0.95)
        fig = self.figures["figureFeatureImportanceBar"]

        maksimi = 0
        column = "relativeImportance"
                       
        for ind,trainingSet in enumerate(list(self.featureImportanceDataCollection)):
            if self.featureImportanceDataCollection[trainingSet] is None:
                continue
            
            ax = fig.getAxes(ind)
            
            for tt in list(self.featureImportanceDataCollection[trainingSet]):
                if "_Emulated" in tt:
                    predictor =  tt
            
            dataframe = self.featureImportanceDataCollection[trainingSet][predictor]
            maksimi = max(dataframe[column].max(), maksimi)
            
            dataframe.plot(ax=ax, kind="bar",color=dataframe["color"], x="mathLabel", y = column, legend = False)
            PlotTweak.setAnnotation(ax, f"{self.annotationValues[ind]}",
                                    xPosition=ax.get_xlim()[1]*0.07, yPosition = 0.4)
        
        for ind in range(nrows*ncols):
            ax = fig.getAxes(ind)
            
            ymax = round(maksimi, 1) + 0.11
            yTicks = numpy.arange(0, ymax, 0.1)
            yTickLabels = [f"{t:.1f}" for t in yTicks]
            showList = Data.cycleBoolean(len(yTicks))
            ax.set_yticks(yTicks)
            ax.set_yticklabels(yTickLabels)
            ax.set_ylim([0, ymax])
            PlotTweak.hideLabels(ax.yaxis, showList)
            
            PlotTweak.setXaxisLabel(ax,"")
            
            
            if ind not in numpy.asarray(range(nrows))*ncols:
                PlotTweak.hideYTickLabels(ax)
            
            if ind == 1:
                
                PlotTweak.setAnnotation(ax, "Gaussian Process Updraft Emulator", xPosition=-0.7, yPosition = 1.05,  xycoords = "axes fraction")
            

        fig.getAxes(ind).legend(handles=PlotTweak.getPatches(self.mathLabelColors),
                      loc=(-0.9,-0.6),
                      ncol = self.legendCols,
                      fontsize = 8)

    
    def figureWposPredictorsVsSimulated(self):
        
        self._figurePredictorsVsSimulated("wpos", 0.8)
    
    def figureW2posPredictorsVsSimulated(self):
        self._figurePredictorsVsSimulated("w2pos", 1.0)
    
    def _figurePredictorsVsSimulated(self, name, end):
        ncols = 2
        nrows = 2
        figureName = f"figure{name}PredictorsVsSimulated"
        self.figures[figureName] = Figure(self.figureFolder,figureName,
                                                   figsize = [self.figureWidth, 4.5],  ncols = ncols, nrows = nrows,
                                                   bottom = 0.15, hspace = 0.2, wspace=0.15, top=0.95, left=0.14, right = 0.97)
        fig = self.figures[figureName]
        
        start = 0
        ticks = numpy.arange(0, end + 0.1, 0.1)
        tickLabels = [f"{t:.1f}" for t in ticks]

        showList = Data.cycleBoolean(len(ticks))

        showList[0] = False
        # showList[-1] = False
        col = -1
        ratioColor = Colorful.getDistinctColorList("red")
        fitColor = "k"
        for ind,trainingSet in enumerate(self.trainingSetList):
        
            ax = fig.getAxes(ind)
            shortname = self.predictorShortNames[-1]
            dataframe = copy.deepcopy(self.completeDataFrame[trainingSet])
            
            dataframe[self.responseVariable] = dataframe[self.responseVariable] 
            dataframe[self.emulatedVariable] = dataframe[self.emulatedVariable] 
            if self.completeDataFrame[trainingSet] is None or self.statsCollection[trainingSet] is None:
                continue

            dataframe = dataframe.loc[dataframe[self.filterIndex]]

            simulated = dataframe[self.responseVariable]

            statistics = self.statsCollection[trainingSet].loc[ self.predictorShortNames[col] ]

            slope = statistics["slope"]
            intercept = statistics["intercept"]
            rSquared = statistics["rSquared"]
            rmse = statistics["rmse"]
            
            

            dataframe.plot.scatter(ax = ax, x=self.responseVariable, y=self.emulatedVariable,color = self.predictorColors[ shortname ], alpha=0.3)

            coef = [slope, intercept]
            poly1d_fn = numpy.poly1d(coef)
            ax.plot(simulated.values, poly1d_fn(simulated.values), color = fitColor)

            ax.plot([0,1e9],[0,1e9], color=ratioColor)
            
            PlotTweak.setAnnotation(ax, f"""{PlotTweak.getLatexLabel(f'R^2={rSquared:.2f}','')}
{PlotTweak.getLatexLabel(f'RMSE={rmse:.3f}','')}""",
                                    xPosition=0.12, yPosition=0.70, bbox_props = None, xycoords="axes fraction")
            
            PlotTweak.setXaxisLabel(ax,"")
            PlotTweak.setYaxisLabel(ax,"")
                
            ax.set_ylim([start, end])
            ax.set_xlim([start, end])
            
            ax.set_xticks(ticks)
            ax.set_xticklabels(tickLabels)
            PlotTweak.hideLabels(ax.xaxis, showList)
            
            PlotTweak.hideLabels(ax.yaxis, showList)
            
            PlotTweak.setXTickSizes(ax, Data.cycleBoolean(len(ticks)))
            PlotTweak.setYTickSizes(ax, Data.cycleBoolean(len(ticks)))
    
            
        for ind,trainingSet in enumerate(self.trainingSetList):
            ax = fig.getAxes(ind)
            
            PlotTweak.setAnnotation(ax, f"{self.annotationValues[ind]}",
                                        xPosition=ax.get_xlim()[1]*0.05, yPosition = ax.get_ylim()[1]*0.90)
            
            if ind in [1,3]:
                PlotTweak.hideYTickLabels(ax)
            else:
                ax.text(PlotTweak.getXPosition(ax, -0.56), PlotTweak.getYPosition(ax, 0.3),
                        PlotTweak.getLatexLabel(self.trainingSetSensibleNames[ind//3]), size=8 , rotation =90)
                
            if ind == 1:
                PlotTweak.setAnnotation(ax, f"Gaussian Process Updraft Emulator ({name})", xPosition=-0.7, yPosition = 1.05,  xycoords = "axes fraction")
                
            if ind == 2:
                ax.text(PlotTweak.getXPosition(ax, -0.32), PlotTweak.getYPosition(ax, 0.5),
                        PlotTweak.getUnitLabel("Predicted\ Updraft", PlotTweak.getVariableUnit("wpos")), size=8 , rotation =90)
            if ind == 3:
                ax.text(PlotTweak.getXPosition(ax, -0.7), PlotTweak.getYPosition(ax, -0.22),PlotTweak.getUnitLabel("Simulated\ Updraft", PlotTweak.getVariableUnit("wpos")) , size=8)
                
                legendLabelColors = PlotTweak.getPatches( dict(zip(["1:1 ratio","Lin. Reg. Fit"], [ratioColor, fitColor])))

                artist = ax.legend( handles=legendLabelColors, loc=(-0.7, -0.37), frameon = True, framealpha = 1.0, ncol = 2 )
            
                ax.add_artist(artist)

    
    def figureUpdraftDistributions(self):
        nrows = 2
        ncols = 2
        
        self.figures["figureUpdraftDistributions"] = Figure(self.figureFolder,"figureUpdraftDistributions",
                                                   figsize = [self.figureWidth, 7],  ncols = ncols, nrows = nrows,
                                                   bottom = 0.08, hspace = 0.17, wspace=0.35, top=0.95, left=0.12, right = 0.95)
        fig = self.figures["figureUpdraftDistributions"]
                
        
        minisDesigns = [numpy.nan]*len(self.designVariablePool)
        maxisDesigns = [numpy.nan]*len(self.designVariablePool)
        
        echamColor = Colorful.getDistinctColorList("yellow")
        LEScolor = Colorful.getDistinctColorList("orange")
        emulatorColor = Colorful.getDistinctColorList("red")
        
        start = 0
        end = 1.4
        ticks = numpy.arange(0, end + 0.01, 0.1)
        tickLabels = [f"{t:.1f}" for t in ticks]

        showList = Data.cycleBoolean(len(ticks))

        showList[0] = False
        showList[-1] = True
        
        yticks = numpy.arange(0, 8+ 0.5, 1)
        print(yticks)
        yTickLabels = [f"{t:.0f}" for t in yticks]
        print(yTickLabels)
                
        for ind,trainingSet in enumerate(self.trainingSetList):
            ax = fig.getAxes(ind)
            
            minimi = numpy.nan
            maximi = numpy.nan
            
            
            if hasattr(self, "filteredSourceData") and (self.filteredSourceData is not None):
                                
                print("all echam columns: ", len(self.filteredSourceData["W"]))
                
                if "Day" in trainingSet:
                    filteredSourceData_TimeOfDay = self.filteredSourceData[self.filteredSourceData["cos_mu"] > Data.getEpsilon()]
                    timeOfDay = "Day"
                else:
                    filteredSourceData_TimeOfDay = self.filteredSourceData[self.filteredSourceData["cos_mu"] < Data.getEpsilon()]
                    timeOfDay = "Night"
                
                sourceDataVariable = filteredSourceData_TimeOfDay["W"]
                print(f'{timeOfDay} {len(filteredSourceData_TimeOfDay)/len(self.filteredSourceData["W"]):.2f}' )
                
                sourceDataVariable = Data.dropInfNanFromDataFrame(sourceDataVariable)
                
                sourceDataVariable.plot.density(ax = ax, color =echamColor)
                maximi = numpy.nanmax([maximi, sourceDataVariable.max()])
                
            
            dataframe = self.completeDataFrame[trainingSet]
            dataframe = dataframe.loc[dataframe[self.filterIndex]]
            
            simulated = dataframe[self.responseVariable] 
            emulated = dataframe[self.emulatedVariable] 
            
            print(trainingSet, simulated.max(), emulated.max())
                    
            simulated.plot.density(ax = ax, color = LEScolor  )
            emulated.plot.density(ax = ax, color = emulatorColor )
            
            
            maximi = numpy.nanmax([maximi, simulated.max()])
            maximi = numpy.nanmax([maximi, emulated.max()])
            
            variableAnnotationYposition = 0.9
            
            
                
            annotation = f"{self.annotationValues[ind]}"
            PlotTweak.setAnnotation(ax, annotation,
                                    xPosition =  0.2,
                                    yPosition = variableAnnotationYposition,
                                    xycoords = "axes fraction") 
            ax.set_ylabel("")
            ax.set_xlabel("")
            
            
            
            ax.set_xticks(ticks)
            ax.set_xticklabels(tickLabels)
            
            PlotTweak.hideLabels(ax.xaxis, showList)
            PlotTweak.setXTickSizes(ax, Data.cycleBoolean(len(ticks)))
            ax.set_xlim([0, ticks[-1]])
            # ax.set_xlim([0, maximi])    
            
            ax.set_yticks(yticks)
            ax.set_yticklabels(yTickLabels)
            # PlotTweak.setYTickSizes(ax, Data.cycleBoolean(len(yticks)))
                
        
        legendLabelColors = PlotTweak.getPatches( dict(zip(["ECHAM","LES", "Emulator"], [echamColor, LEScolor, emulatorColor])))

        artist = ax.legend( handles=legendLabelColors, loc=(-0.9, -0.18), frameon = True, framealpha = 1.0, ncol = 3 )

        ax.add_artist(artist)
        
        for ind,variable in enumerate(self.trainingSetList):
            ax = fig.getAxes(ind)
            
            ax.set_ylim([0,ax.get_ylim()[1]])
            
            if ind == 1:
                PlotTweak.setAnnotation(ax, PlotTweak.getUnitLabel("Updraft\ Distributions", PlotTweak.getVariableUnit("wpos")), xPosition=-0.9, yPosition = 1.05,  xycoords = "axes fraction")
            

def main():
    try:
        locationsFile = sys.argv[1]
    except IndexError:
        locationsFile = "/home/aholaj/mounttauskansiot/puhtiwork/UpdraftEmulator/updraftLocationsMounted.yaml"
        
    figObject = ManuscriptFigures(locationsFile)

    if True:
        figObject.initReadFeatureImportanceData()
        figObject.figureBarFeatureImportanceData()
        
        
    if True:
        figObject.figureWposPredictorsVsSimulated()
    
    
    if False:
        figObject.figureW2posPredictorsVsSimulated()
    
    if False:
        figObject.initReadFilteredSourceData()
        figObject.figureUpdraftDistributions()
        
    figObject.finalise()

if __name__ == "__main__":
    start = time.time()
    now = datetime.now().strftime('%d.%m.%Y %H.%M')
    print(f"Script started {now}.")
    main()
    end = time.time()
    now = datetime.now().strftime('%d.%m.%Y %H.%M')
    print(f"Script completed {now} in {Data.timeDuration(end - start)}")
