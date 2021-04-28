#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 15:46:01 2020

@author: Jaakko Ahola, Finnish Meteorological Institute
@licence: MIT licence Copyright

Plot figures for Updraft emulator manuscript
"""
print(__doc__)
import matplotlib
import math
import numpy
import os
import pandas
import pathlib
import sys
from scipy import stats
import time

sys.path.append(os.environ["LESMAINSCRIPTS"])
from Colorful import Colorful
from Data import Data
from Figure import Figure
from PlotTweak import PlotTweak

from sklearn.metrics import mean_squared_error
from math import sqrt

sys.path.append("../LES-emulator-02postpros")
from EmulatorMetaData import EmulatorMetaData

class ManuscriptFigures(EmulatorMetaData):

    def __init__(self, emulatorPostprosDataRootFolder, figureFolder, configFile):
        
        self.responseIndicatorVariable = "responseIndicator"
        
        super().__init__(configFile)
        
        self.figures = {}
        
    
        self.trainingSetList = ["LVL3Night",
                               "LVL3Day",
                               "LVL4Night",
                               "LVL4Day"]

        self.emulatorPostprosDataRootFolder = pathlib.Path(emulatorPostprosDataRootFolder)
        self.figureFolder = pathlib.Path(figureFolder)

        self.figureFolder.mkdir( parents=True, exist_ok = True )



        self.annotationValues = ["(a) SB Night",
            "(b) SB Day",
            "(c) SALSA Night",
            "(d) SALSA Day"]
        
        self.traininSetSensibleNames = ["SB\ Night",
            "SB\ Day",
            "SALSA\ Night",
            "SALSA\ Day"]

        self.annotationCollection = dict(zip(self.trainingSetList, self.annotationValues))

        self.cloudTopColor = Colorful.getDistinctColorList("green")
        self.lwpColor = Colorful.getDistinctColorList("blue")
        self.tempColor = Colorful.getDistinctColorList("yellow")

        self.observationParameters = {}
        self.observationParameters["slope"] = -0.44/100.
        self.observationParameters["intercept"] = 22.30/100.
        self.observationParameters["error"] = 13./100.



        self._initReadCompleteData()

        self._filterCompleteData()

        self._initReadStats()

        self._initReadLimits()

    def _initReadCompleteData(self):
        self.completeDataFrame = {}
        for trainingSet in self.trainingSetList:
            self.completeDataFrame[trainingSet] = pandas.read_csv( self.emulatorPostprosDataRootFolder / trainingSet / ( trainingSet + "_complete.csv" ), index_col = 0  )

    def _filterCompleteData(self):
        self.completeDataFrameFiltered = {}
        for trainingSet in self.trainingSetList:
            dataframe = self.completeDataFrame[trainingSet]
            dataframe = dataframe.loc[ dataframe[self.filterIndex] ]
            self.completeDataFrameFiltered[trainingSet] = dataframe

    def initReadFeatureImportanceData(self):
        self.featureImportanceDataCollection = {}
        self.allLabels = []
        self.legendCols = 4
        for trainingSet in self.trainingSetList:
            self.featureImportanceDataCollection[trainingSet] = {}
            dataset = pandas.read_csv( self.emulatorPostprosDataRootFolder / trainingSet / ( trainingSet + "_featureImportance.csv" ), index_col = 0 )
            for ind in dataset.index:
                series = dataset.loc[ind]
                mean = series.loc[[kk for kk in series.index if kk[-4:] == "Mean"]]
                std = series.loc[[kk for kk in series.index if kk[-4:] == "Mean"]]
                ines = [kk[:-22] for kk in series.index if kk[-4:] == "Mean"]
                
                self.allLabels += ines
                
                subset = pandas.DataFrame(data =  {"Mean" : mean.values, "Std": std.values}, index = ines).dropna()
                
                subset["mathLabel"] = subset.apply(lambda row: PlotTweak.getMathLabel(row.name), axis = 1)

                subset = subset.sort_values("Mean", ascending=False)
                subset["points"] = range(len(subset))
                summa = subset["Mean"].sum()

                subset["relativeImportance"] = subset.apply(lambda row: max(row["Mean"], 0) / summa, axis = 1)
                
                self.featureImportanceDataCollection[trainingSet][ind] = subset
        
        self.uniqueLabels = list(set(self.allLabels))
        
        ###
        self.labelPoints = dict(zip(self.uniqueLabels, [0]*len(self.uniqueLabels)))
        for key in self.uniqueLabels:
            for row,trainingSet in enumerate(list(self.featureImportanceDataCollection)):
                for col, predictor in enumerate(list(self.featureImportanceDataCollection[trainingSet])):
                    dataframe = self.featureImportanceDataCollection[trainingSet][predictor]
                    
                    if key in dataframe.index:
                        self.labelPoints[key] += dataframe.loc[key]["points"]
                        
        self.labelPoints = pandas.Series(self.labelPoints).sort_values().to_dict()
        ###
        self.labelRelative = dict(zip(self.uniqueLabels, [1]*len(self.uniqueLabels)))
        
        self.zeros = dict(zip(self.uniqueLabels, [0]*len(self.uniqueLabels)))
        for key in self.uniqueLabels:
            for row,trainingSet in enumerate(list(self.featureImportanceDataCollection)):
                for col, predictor in enumerate(list(self.featureImportanceDataCollection[trainingSet])):
                    dataframe = self.featureImportanceDataCollection[trainingSet][predictor]
                    
                    if key in dataframe.index:
                        relative = dataframe.loc[key]["relativeImportance"]
                        if relative < Data.getEpsilon():
                            relative = 1
                            self.zeros[key] += 1
                        self.labelRelative[key] *= relative
        
        ###
        relativeCombined = pandas.DataFrame.from_dict(self.labelRelative, orient="index", columns = ["relativeCombined"])
        zeros = pandas.DataFrame.from_dict(self.zeros, orient="index", columns = ["zeros"])
        labelOrder = pandas.concat((relativeCombined, zeros), axis = 1)
        
        self.labelRelative = []
        self.labelCategorised = []
        for zeroAmount in set(labelOrder["zeros"].values):
            subdf = labelOrder[labelOrder["zeros"] == zeroAmount].sort_values(by="relativeCombined", ascending = False)
            self.labelCategorised.append(subdf)
            self.labelRelative += list(subdf.index.values)
        
        ###
        
        self.labelCategorised = pandas.concat(self.labelCategorised)
        
        self.labelCategorised["mathLabel"] = self.labelCategorised.apply(lambda row: PlotTweak.getMathLabel(row.name), axis = 1)
        
        # print(self.labelCategorised)
        
        ###
        
        self.uniqueLabels = self.labelRelative
        
        self._getColorsForLabels()
        
        for trainingSet in self.featureImportanceDataCollection:
            for predictor in self.featureImportanceDataCollection[trainingSet]:
                subset  = self.featureImportanceDataCollection[trainingSet][predictor]
                subset["color"] = subset.apply(lambda row: self.labelColors[row.name], axis = 1)
        
        

    def _initReadStats(self):
        self.statsCollection = {}
        for trainingSet in self.trainingSetList:
            self.statsCollection[trainingSet] = pandas.read_csv( self.emulatorPostprosDataRootFolder / trainingSet / ( trainingSet + "_stats.csv" ), index_col = 0  )
    def _initReadLimits(self):
        self.anomalyLimits = pandas.read_csv( self.emulatorPostprosDataRootFolder / "anomalyLimits.csv", index_col = 0)

    def _getColorsForLabels(self):
        self.uniqueLabels = list(numpy.array(self.uniqueLabels).reshape(-1,self.legendCols).T.reshape(-1,1).ravel())
        
        self.uniqueColors = [ PlotTweak.getLabelColor( label ) for label in self.uniqueLabels ]
        
        
        
        
        self.labelColors = dict(zip(self.uniqueLabels, self.uniqueColors))
        
        self.uniqueMathLabels = [PlotTweak.getMathLabel(label) for label in self.uniqueLabels]
        
        self.mathLabelColors = dict(zip(self.uniqueMathLabels, self.uniqueColors))

    def finalise(self):

        for fig in self.figures.values():
            fig.save()


    def figureBarFeatureImportanceData(self):
        ncols = len(self.featureImportanceDataCollection[list(self.featureImportanceDataCollection)[0]]) # number of methods
        nrows = len(self.featureImportanceDataCollection) # = number of training sets

        self.figures["figureFeatureImportanceBar"] = Figure(self.figureFolder,"figureFeatureImportanceBar",
                                                            figsize=(12/2.54,6), 
                                                            ncols = ncols, nrows = nrows,
                                                            hspace=0.8, bottom=0.20, wspace = 0.05, top = 0.97)
        fig = self.figures["figureFeatureImportanceBar"]

        grey = Colorful.getDistinctColorList("grey")
        
        maksimi = 0
        column = "relativeImportance"
        for row,trainingSet in enumerate(list(self.featureImportanceDataCollection)):
            for col, predictor in enumerate(list(self.featureImportanceDataCollection[trainingSet])):
                
                dataframe = self.featureImportanceDataCollection[trainingSet][predictor]
                
                maksimi = max(dataframe[column].max(), maksimi)
                ax = fig.getAxesGridPoint( {"row": row, "col": col})
                
                dataframe.plot(ax=ax, kind="bar",color=dataframe["color"], x="mathLabel", y = column, legend = False)
                
        
                
        
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
            
            PlotTweak.setAnnotation(ax, f"({Data.getNthLetter(ind)})",
                                    xPosition=ax.get_xlim()[1]*0.07, yPosition = ax.get_ylim()[1]*0.80)
            
            if ind not in numpy.asarray(range(nrows))*ncols:
                PlotTweak.hideYTickLabels(ax)
            else:
                ax.text(PlotTweak.getXPosition(ax, -0.24), PlotTweak.getYPosition(ax, 0.),
                            PlotTweak.getLatexLabel(self.traininSetSensibleNames[ind//ncols]), size=8 , rotation =90)
            if ind == 0:
                ax.text(PlotTweak.getXPosition(ax, 0.2), PlotTweak.getYPosition(ax, 1.05), "Emulator", size=8)
            if ind == 1:
                ax.text(PlotTweak.getXPosition(ax, 0.2),PlotTweak.getYPosition(ax, 1.05), "Corrected linear fit", size=8)

        fig.getAxes(-1).legend(handles=PlotTweak.getPatches(self.mathLabelColors),
                      loc=(-0.9,-1.59),
                      ncol = self.legendCols,
                      fontsize = 8)

    
    
    def figureMethodsVsSimuted(self):
        numberOfMethods = 3
        self.figures["figureMethodsVsSimuted"] = Figure(self.figureFolder,"figureMethodsVsSimuted",
                                                   figsize = [4.724409448818897, 7],  ncols = numberOfMethods, nrows = 4,
                                                   bottom = 0.07, hspace = 0.09, wspace=0.09, top=0.95, left=0.16, right = 0.98)
        fig = self.figures["figureMethodsVsSimuted"]
        
        print("figureMethodsVsSimuted")
        
        start = 0.0
        end = 1.0
        ticks = numpy.arange(0, end + .01, 0.1)
        tickLabels = [f"{t:.1f}" for t in ticks]

        showList = Data.cycleBoolean(len(ticks))

        showList[0] = False
        showList[-1] = False
        
        
                
        
        emulatorColor = Colorful.getDistinctColorList("blue")
        linearColor = Colorful.getDistinctColorList("red")
        correctedColor = Colorful.getDistinctColorList("green")
        
        rSquaredList = numpy.zeros(12)
        
        ncol = 0 # emulator
        for setInd,trainingSet in enumerate(self.trainingSetList):
            ind = ncol + numberOfMethods*setInd
            ax = fig.getAxes(ind)

            dataframe = self.completeDataFrame[trainingSet]

            dataframe = dataframe.loc[dataframe[self.filterIndex]]


            simulated = dataframe[self.responseVariable]

            statistics = self.statsCollection[trainingSet].loc["leaveOneOutStats"]

            slope = statistics["slope"]
            intercept = statistics["intercept"]
            rSquared = statistics["rSquared"]

            rSquaredList[ind] = rSquared

            dataframe.plot.scatter(ax = ax, x=self.responseVariable, y=self.emulatedVariable,color = emulatorColor, alpha=0.3)

            coef = [slope, intercept]
            poly1d_fn = numpy.poly1d(coef)
            ax.plot(simulated.values, poly1d_fn(simulated.values), color = "k")


            

            PlotTweak.setXaxisLabel(ax,"")
            PlotTweak.setYaxisLabel(ax,"")
        
        ncol+=1 # linear fit
        for setInd,trainingSet in enumerate(self.trainingSetList):
            ind = ncol + numberOfMethods*setInd
            ax = fig.getAxes(ind)

            dataframe = self.completeDataFrameFiltered[trainingSet]
            simulatedFit = dataframe[self.responseVariable]
            fittedFit = dataframe[self.linearFitVariable]

            slopeFit, interceptFit, r_valueFit, p_valueFit, std_errFit = stats.linregress(simulatedFit, fittedFit)

            rSquaredFit = numpy.power(r_valueFit,2)
            
            rSquaredList[ind] = rSquaredFit

            coefFit = [slopeFit, interceptFit]

            fitColor = "k"
            
            dataframe.plot.scatter(ax=ax, x=self.responseVariable, y=self.linearFitVariable, alpha = 0.3, color=linearColor)

            poly1d_fn = numpy.poly1d(coefFit)

            ax.plot(simulatedFit.values, poly1d_fn(simulatedFit.values), color = fitColor)
            
        ncol += 1 # corrected linear fit
        for setInd,trainingSet in enumerate(self.trainingSetList):
            ind = ncol + numberOfMethods*setInd
            ax = fig.getAxes(ind)

            dataframe = self.completeDataFrame[trainingSet]

            dataframe = dataframe.loc[dataframe[self.filterIndex]]

            simulated =  dataframe[ self.responseVariable ].values
            
            # statistics = self.statsCollection[trainingSet].loc["correctedLinearFitStats"]

            # slope = statistics["slope"]
            # intercept = statistics["intercept"]
            # rSquared = statistics["rSquared"]
            corrected = dataframe[ self.correctedLinearFitVariable ].values
            simulated =  dataframe[ self.responseVariable ].values
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(simulated, corrected)

            rSquared = numpy.power(r_value, 2)
            
            rSquaredList[ind] = rSquared

            dataframe.plot.scatter(ax = ax, x=self.responseVariable, y=self.correctedLinearFitVariable, alpha=0.3, color = correctedColor)

            coef = [slope, intercept]
            poly1d_fn = numpy.poly1d(coef)
            ax.plot(simulated, poly1d_fn(simulated), color = "k")

    
        for ind in range(12):
            ax = fig.getAxes(ind)
            ax.set_ylim([start, end])

            ax.set_xlim([start, end])

            PlotTweak.setAnnotation(ax, f"({Data.getNthLetter(ind)})",
                                    xPosition=ax.get_xlim()[1]*0.05, yPosition = ax.get_ylim()[1]*0.90)
            
            PlotTweak.setAnnotation(ax, PlotTweak.getLatexLabel(f"R^2={rSquaredList[ind]:.2f}",""), xPosition=0.5, yPosition=0.1, bbox_props = None)
            
            PlotTweak.setXaxisLabel(ax,"")
            PlotTweak.setYaxisLabel(ax,"")
            
            PlotTweak.setXTickSizes(ax, Data.cycleBoolean(len(ticks)))
            PlotTweak.setYTickSizes(ax, Data.cycleBoolean(len(ticks)))
            
            ax.set_xticks(ticks)
            ax.set_xticklabels(tickLabels)
            PlotTweak.hideLabels(ax.xaxis, showList)
            
            ax.set_yticks(ticks)
            ax.set_yticklabels(tickLabels)
            PlotTweak.hideLabels(ax.yaxis, showList)
            
            if ind not in numpy.asarray(range(4))*3:
                PlotTweak.hideYTickLabels(ax)
            else:
                ax.text(PlotTweak.getXPosition(ax, -0.56), PlotTweak.getYPosition(ax, 0.3),
                        PlotTweak.getLatexLabel(self.traininSetSensibleNames[ind//3]), size=8 , rotation =90)

            if ind not in list(range(9,12)):
                PlotTweak.hideXTickLabels(ax)
                
            if ind == 1:
                collectionOfLabelsColors = {"Emulator": emulatorColor, "Linear Fit" : linearColor, "Corr. Lin. Fit" : correctedColor}
                legendLabelColors = PlotTweak.getPatches(collectionOfLabelsColors)

                artist = ax.legend( handles=legendLabelColors, loc=(-.9, 1.05), frameon = True, framealpha = 1.0, ncol = 3 )

                ax.add_artist(artist)
                
            if ind == 3:
                ax.text(PlotTweak.getXPosition(ax, -0.42), PlotTweak.getYPosition(ax, -0.5),
                        PlotTweak.getUnitLabel("Predicted\ w_{pos}", "m\ s^{-1}"), size=8 , rotation =90)
            if ind == 10:
                ax.text(-0.1,-0.27,PlotTweak.getUnitLabel("Simulated\ w_{pos}", "m\ s^{-1}") , size=8)
                
            
    def analyseLinearFit(self):
        condition = {}
        updraftVariableName = self.responseVariable

        data = {}
        

        variables = ["q_inv", "tpot_inv", "lwp", "tpot_pbl", "pblh", "cos_mu", "pblh_m", "prcp", "wpos", "w2pos", "drflx", "lwpEndValue", "lwpRelativeChange", "cfracEndValue", "cloudTopRelativeChange"]

        for column in variables:
            data[column + "_Inside_min"] = []
            data[column + "_Inside_max"] = []
            data[column + "_Outlier_min"] = []
            data[column + "_Outlier_max"] = []


        for ind,trainingSet in enumerate(self.trainingSetList):
            dataframe = self.completeDataFrameFiltered[trainingSet]

            slope, intercept, r_value, p_value, std_err = stats.linregress(dataframe["drflx"], dataframe["w2pos"])

            rSquared = numpy.power(r_value, 2)



            condition["notMatchObservation"] = ~ ( (dataframe[updraftVariableName] > dataframe["drflx"]*self.observationParameters["slope"]+ self.observationParameters["intercept"]-self.observationParameters["error"]) & (dataframe[updraftVariableName] < dataframe["drflx"]*self.observationParameters["slope"]+ self.observationParameters["intercept"]+self.observationParameters["error"]))

            dataFrameInside = dataframe[ ~condition["notMatchObservation"]]

            slopeInside, interceptInside, r_valueInside, p_valueInside, std_errInside = stats.linregress(dataFrameInside["drflx"], dataFrameInside["w2pos"])

            rSquaredInside = numpy.power(r_valueInside, 2)

            print(f"{trainingSet}, all R^2: {rSquared:.2f}, #: {dataframe.shape[0]}; inside R^2: {rSquaredInside:.2f} #: {dataFrameInside.shape[0]}")

            for column in variables:

                try:
                    minimiOutlier = float(dataframe.loc[ condition["notMatchObservation"] ][column].min())
                except (TypeError, ValueError, KeyError):
                    minimiOutlier = -8888.
                try:
                    maksimiOutlier = float(dataframe.loc[ condition["notMatchObservation"] ][column].max())
                except (TypeError, ValueError, KeyError):
                    maksimiOutlier = -8888.

                try:
                    minimiInside = float(dataframe.loc[ ~ condition["notMatchObservation"] ][column].min())
                except (TypeError, ValueError, KeyError):
                    minimiInside = -8888.
                try:
                    maksimiInside = float(dataframe.loc[~ condition["notMatchObservation"] ][column].max())
                except (TypeError, ValueError, KeyError):
                    maksimiInside = -8888.

                data[ column + "_Inside_min"].append(minimiInside)
                data[ column + "_Inside_max"].append(maksimiInside)

                data[ column + "_Outlier_min"].append(minimiOutlier)
                data[ column + "_Outlier_max"].append(maksimiOutlier)

                if minimiOutlier < minimiInside:
                    print(f"{trainingSet} {column} minimi smaller outside")
                if maksimiOutlier > maksimiInside:
                    print(f"{trainingSet} {column} maksimi greater outside")


                # print(f"{trainingSet:11}{column:33} Inside: {minimiInside:.2f}{maksimiInside:.2f} Outlier: {minimiOutlier:.2f}{maksimiOutlier:.2f}")
        df = pandas.DataFrame(data, index = self.trainingSetList)

        df.to_csv("/home/aholaj/Data/EmulatorManuscriptDataW2Pos/analyseLinearfit.csv")

    def analyseLinearFitPercentile(self):
        condition = {}
        updraftVariableName = self.responseVariable

        data = {}

        variables = ["q_inv", "tpot_inv", "lwp", "tpot_pbl", "pblh", "cos_mu", "pblh_m", "prcp", "wpos", "w2pos", "drflx", "lwpEndValue", "lwpRelativeChange", "cfracEndValue", "cloudTopRelativeChange"]
        tailPercentile = 0.05
        for column in variables:
            data[column + "_Inside_low_tail"] = []
            data[column + "_Inside_high_tail"] = []
            data[column + "_Outlier_low_tail"] = []
            data[column + "_Outlier_high_tail"] = []

        print("Percentile")
        for ind,trainingSet in enumerate(self.trainingSetList):
            dataframe = self.completeDataFrameFiltered[trainingSet]
            condition["notMatchObservation"] = ~ ( (dataframe[updraftVariableName] > dataframe["drflx"]*self.observationParameters["slope"]+ self.observationParameters["intercept"]-self.observationParameters["error"]) & (dataframe[updraftVariableName] < dataframe["drflx"]*self.observationParameters["slope"]+ self.observationParameters["intercept"]+self.observationParameters["error"]))
            for column in variables:

                try:
                    minimiOutlier = float(dataframe.loc[ condition["notMatchObservation"] ][column].quantile( tailPercentile ))
                except (TypeError, ValueError, KeyError):
                    minimiOutlier = -8888.
                try:
                    maksimiOutlier = float(dataframe.loc[ condition["notMatchObservation"] ][column].quantile(1- tailPercentile ))
                except (TypeError, ValueError, KeyError):
                    maksimiOutlier = -8888.

                try:
                    minimiInside = float(dataframe.loc[ ~ condition["notMatchObservation"] ][column].quantile( tailPercentile ))
                except (TypeError, ValueError, KeyError):
                    minimiInside = -8888.
                try:
                    maksimiInside = float(dataframe.loc[~ condition["notMatchObservation"] ][column].quantile(1- tailPercentile ))
                except (TypeError, ValueError, KeyError):
                    maksimiInside = -8888.

                data[ column + "_Inside_low_tail"].append(minimiInside)
                data[ column + "_Inside_high_tail"].append(maksimiInside)

                data[ column + "_Outlier_low_tail"].append(minimiOutlier)
                data[ column + "_Outlier_high_tail"].append(maksimiOutlier)

                if minimiOutlier < minimiInside:
                    print(f"{trainingSet} {column} low tail smaller outside")
                if maksimiOutlier > maksimiInside:
                    print(f"{trainingSet} {column} high tail greater outside")


                # print(f"{trainingSet:11}{column:33} Inside: {minimiInside:.2f}{maksimiInside:.2f} Outlier: {minimiOutlier:.2f}{maksimiOutlier:.2f}")
        df = pandas.DataFrame(data, index = self.trainingSetList)

        df.to_csv("/home/aholaj/Data/EmulatorManuscriptDataW2Pos/analyseLinearfitPercentile.csv")

    def figureUpdraftLinearFit(self):


        self.figures["figureLinearFit"] = Figure(self.figureFolder,"figureLinearFit", figsize = [4.724409448818897, 4],
                                                 ncols = 2, nrows = 2, bottom = 0.11, hspace = 0.08, wspace=0.12, top=0.95)
        fig = self.figures["figureLinearFit"]

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

            dataframe = self.completeDataFrameFiltered[trainingSet]

            updraftVariableName = self.responseVariable

            radiativeWarming  = dataframe["drflx"].values
            updraft =  dataframe[updraftVariableName].values


            poly1d_Observation = numpy.poly1d(numpy.asarray([self.observationParameters["slope"],self.observationParameters["intercept"] ])) #?0.44 ×CTRC+
            ax.plot(radiativeWarming, poly1d_Observation(radiativeWarming), color = color_obs)
            ax.fill_between(sorted(radiativeWarming),
                            poly1d_Observation(sorted(radiativeWarming)) - self.observationParameters["error"]*numpy.ones(numpy.shape(radiativeWarming)), poly1d_Observation(sorted(radiativeWarming)) + self.observationParameters["error"]*numpy.ones(numpy.shape(radiativeWarming)),
                            alpha=0.2)

            slope, intercept, r_value, p_value, std_err = stats.linregress(radiativeWarming, updraft)
            coef = [slope, intercept]
            rSquared = numpy.power(r_value, 2)

            dataColor = Colorful.getDistinctColorList("red")
            fitColor = "k"
            dataframe.plot.scatter(ax = ax, x="drflx", y=updraftVariableName, alpha=0.3, color = dataColor)

            poly1d_fn = numpy.poly1d(coef)

            linearFit = []
            for radWarmingValue in list(self.completeDataFrame[trainingSet]["drflx"]):
                linearFit.append(poly1d_fn(radWarmingValue))

            self.completeDataFrame[trainingSet][self.linearFitVariable] = linearFit
            ax.plot(radiativeWarming, poly1d_fn(radiativeWarming), color = fitColor)

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

                artist = ax.legend( handles=legendLabelColors, loc=(0.17, 1.02), frameon = True, framealpha = 1.0, ncol = 3 )

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
                
    def figureUpdraftCorrectedLinearFit(self):

        self.figures["figureCorrectedLinearFit"] = Figure(self.figureFolder,"figureCorrectedLinearFit",
                                                          figsize = [4.724409448818897, 4],  ncols = 2, nrows = 2,
                                                   bottom = 0.12, hspace = 0.08, wspace=0.04, top=0.95)
        fig = self.figures["figureCorrectedLinearFit"]
        
        end = 1.0
        ticks = numpy.arange(0, end + .01, 0.1)
        tickLabels = [f"{t:.1f}" for t in ticks]

        showList = Data.cycleBoolean(len(ticks))

        showList[-1] = False
        
        correctedColor = Colorful.getDistinctColorList("green")

        for ind,trainingSet in enumerate(self.trainingSetList):
            ax = fig.getAxes(ind)

            dataframe = self.completeDataFrame[trainingSet]

            dataframe = dataframe.loc[dataframe[self.filterIndex]]


            # statistics = self.statsCollection[trainingSet].loc["correctedLinearFitStats"]

            # slope = statistics["slope"]
            # intercept = statistics["intercept"]
            # rSquared = statistics["rSquared"]

            corrected = dataframe[ self.correctedLinearFitVariable ].values
            simulated =  dataframe[ self.responseVariable ].values
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(simulated, corrected)

            rSquared = numpy.power(r_value, 2)


            dataframe.plot.scatter(ax = ax, x=self.responseVariable, y=self.correctedLinearFitVariable, alpha=0.3, color = correctedColor)


            coef = [slope, intercept]
            poly1d_fn = numpy.poly1d(coef)
            ax.plot(simulated, poly1d_fn(simulated), color = "k")

            ax.set_ylim([0, end])

            ax.set_xlim([0, end])


            PlotTweak.setAnnotation(ax, self.annotationCollection[trainingSet],
                                    xPosition=ax.get_xlim()[1]*0.05, yPosition = ax.get_ylim()[1]*0.90)

            PlotTweak.setAnnotation(ax, PlotTweak.getLatexLabel(f"R^2={rSquared:.2f}",""), xPosition=0.5, yPosition=0.1, bbox_props = None)

            PlotTweak.setXaxisLabel(ax,"")
            PlotTweak.setYaxisLabel(ax,"")

            if ind == 0:
                legendLabelColors = []

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
                ax.text(0.5,-0.25, PlotTweak.getUnitLabel("Simulated\ w_{pos}", "m\ s^{-1}"), size=8)
            if ind == 0:
                ax.text(-0.25,-0.5, PlotTweak.getUnitLabel("Corrected\ Linear\ Fit\ w_{pos}", "m\ s^{-1}"), size=8 , rotation =90)
        
        

    def figureUpdraftLinearFitVSEMul(self):
        self.figures["figureLinearFitComparison"] = Figure(self.figureFolder,"figureLinearFitComparison", figsize = [4.724409448818897, 4.5],
                     ncols = 2, nrows = 2, bottom = 0.11, hspace = 0.08, wspace=0.12, top=0.86)
        fig = self.figures["figureLinearFitComparison"]
        # xticks = numpy.arange(0, xend + 1, 10)

        start = 0.0
        end = 1.0
        ticks = numpy.arange(0, end + .01, 0.1)
        tickLabels = [f"{t:.1f}" for t in ticks]

        showList = Data.cycleBoolean(len(ticks))

        showList[-1] = False

        for ind,trainingSet in enumerate(self.trainingSetList):
            ax = fig.getAxes(ind)

            dataframe = self.completeDataFrameFiltered[trainingSet]
            simulatedFit = dataframe[self.responseVariable]
            fittedFit = dataframe[self.linearFitVariable]

            slopeFit, interceptFit, r_valueFit, p_valueFit, std_errFit = stats.linregress(simulatedFit, fittedFit)

            rSquaredFit = numpy.power(r_valueFit,2)


            coefFit = [slopeFit, interceptFit]


            fitColor = "k"
            dataColor = Colorful.getDistinctColorList("red")

            tempAnomalies = dataframe.loc[dataframe["tpot_inv_low_tail"]]
            cloudTopAnomalies = dataframe.loc[dataframe["cloudTopRelativeChange_high_tail"]]
            lwpAnomalies = dataframe.loc[dataframe["lwpRelativeChange_high_tail"]]

            dataframe.plot.scatter(ax=ax, x=self.responseVariable, y=self.linearFitVariable, alpha = 0.3, color=dataColor)

            dataframeAnomalies = cloudTopAnomalies
            dataframeAnomalies.plot.scatter(ax = ax, x=self.responseVariable, y=self.linearFitVariable, color = self.cloudTopColor, marker = "x", linewidth = 1)

            dataframeAnomalies = tempAnomalies
            dataframeAnomalies.plot.scatter(ax = ax, x=self.responseVariable, y=self.linearFitVariable, color = self.tempColor, marker = "|", linewidth = 1)

            dataframeAnomalies = lwpAnomalies
            dataframeAnomalies.plot.scatter(ax = ax, x=self.responseVariable, y=self.linearFitVariable, color = self.lwpColor, marker = "_", linewidth = 1)

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
                          label='Cloud top rel. change >' + str(self.anomalyLimits.loc["cloudTopRelativeChange"]["high"])))
                legendLabelColors.append(matplotlib.lines.Line2D([], [], color=self.lwpColor, marker='_', markersize = 12, linestyle="None",
                          label='LWP rel. change >' + str(self.anomalyLimits.loc["lwpRelativeChange"]["high"])))
                legendLabelColors.append(matplotlib.lines.Line2D([], [], color=self.tempColor, marker='|', markersize = 12, linestyle='None',
                          label=r"$\Delta {\theta_{L}} < $" + str(self.anomalyLimits.loc["tpot_inv"]["low"])))
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

    def figureDistributionOfUpdrafts(self):
        self.figures["figureDistributionOfUpdrafts"] = Figure(self.figureFolder, "figureDistributionOfUpdrafts", ncols = 2, nrows = 2)

        fig = self.figures["figureDistributionOfUpdrafts"]

        for ind, trainingSet in enumerate(self.trainingSetList):
            ax = fig.getAxes(ind)

            dataframe = self.completeDataFrameFiltered[trainingSet]


            dataframe[self.responseVariable].plot.hist(ax=ax, bins = 20, color = "r",  style='--', alpha = 0.5 )

    def figureWposVSWposWeighted(self):
        self.figures["figureWposVSWposWeighted"] = Figure(self.figureFolder, "figureWposVSWposWeighted", ncols = 2, nrows = 2)

        fig = self.figures["figureWposVSWposWeighted"]

        for ind, trainingSet in enumerate(self.trainingSetList):
            ax = fig.getAxes(ind)

            dataframe = self.completeDataFrame[trainingSet]

            dataframe.plot.scatter(ax=ax, x = "wpos", y="wposWeighted")


    def figureErrorDistribution(self):
        self.figures["figureErrorDistribution"] = Figure(self.figureFolder,"figureErrorDistribution",  ncols = 2, nrows = 2,
                     bottom = 0.15, hspace = 0.08, wspace=0.04, top=0.90)
        fig = self.figures["figureErrorDistribution"]

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

        emulColor = Colorful.getDistinctColorList("blue")
        linFitColor = Colorful.getDistinctColorList("red")
        for ind,trainingSet in enumerate(self.trainingSetList):
            ax = fig.getAxes(ind)

            dataframe = self.completeDataFrameFiltered[trainingSet]

            emulRMSE = sqrt(mean_squared_error(dataframe[self.responseVariable], dataframe[self.emulatedVariable]))


            dataframe["absErrorEmul"].plot.hist(ax=ax, bins = 20, color = emulColor,  style='--', alpha = 0.5 )

            if mini is None:
                mini = dataframe["absErrorEmul"].min()
            else:
                mini = min(mini, dataframe["absErrorEmul"].min())

            if maxi is None:
                maxi = dataframe["absErrorEmul"].max()
            else:
                maxi = max(maxi,  dataframe["absErrorEmul"].min())


            linfitRMSE = sqrt(mean_squared_error(dataframe[self.responseVariable], dataframe[self.linearFitVariable]))

            dataframe["absErrorLinearFit"].plot.hist(ax=ax, bins = 20, color = linFitColor, style='--', alpha = 0.5 )

            mini = min(mini, dataframe["absErrorLinearFit"].min())
            maxi = max(maxi, dataframe["absErrorLinearFit"].max())

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

    def tables_featureImportanceOrder(self):
        self.labelCategorised.to_latex(self.figureFolder / "featureImportanceOrder.tex", columns =["mathLabel", "relativeCombined", "zeros"], index =False)

def main():

    figObject = ManuscriptFigures("/home/aholaj/mounttauskansiot/puhtiwork/EmulatorManuscriptData",
                                  "/home/aholaj/Nextcloud/000_WORK/000_ARTIKKELIT/02_LES-Emulator/001_Manuscript_LES_emulator/figures",
                                  "/home/aholaj/mounttauskansiot/puhtiwork/EmulatorManuscriptData/phase02.yaml")

    if True:
        figObject.initReadFeatureImportanceData()
        figObject.figureBarFeatureImportanceData()
        figObject.tables_featureImportanceOrder()
    if False:
        figObject.figureUpdraftLinearFit()
    if False:
        figObject.figureUpdraftCorrectedLinearFit()
    if False:
        figObject.figureUpdraftLinearFitVSEMul()
    if False:
        figObject.figureErrorDistribution()
    if False:
        figObject.figureDistributionOfUpdrafts()
    if False:
        figObject.figureWposVSWposWeighted()
    if False:
        figObject.figureMethodsVsSimuted()

    figObject.finalise()

    # figObject.analyseLinearFitPercentile()
    # figObject.analyseLinearFit()

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("Script completed in " + str(round((end - start),0)) + " seconds")
