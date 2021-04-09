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
import scipy
import sys
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

    def initReadSensitivityData(self):
        self.sensitivityDataCollection = {}
        for trainingSet in self.trainingSetList:
            self.sensitivityDataCollection[trainingSet] = pandas.read_csv( self.emulatorPostprosDataRootFolder / trainingSet / ( trainingSet + "_sensitivityAnalysis.csv" ), index_col = 0 )

    def _initReadStats(self):
        self.statsCollection = {}
        for trainingSet in self.trainingSetList:
            self.statsCollection[trainingSet] = pandas.read_csv( self.emulatorPostprosDataRootFolder / trainingSet / ( trainingSet + "_stats.csv" ), index_col = 0  )
    def _initReadLimits(self):
        self.anomalyLimits = pandas.read_csv( self.emulatorPostprosDataRootFolder / "anomalyLimits.csv", index_col = 0)

    def _getColorsForLabels(self, labels):
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

    def finalise(self):

        for fig in self.figures.values():
            fig.save()


    def figureBarSensitivyData(self):


        self.figures["figureSensitivityBar"] = Figure(self.figureFolder,"figureSensitivityBar", figsize=(12/2.54,6),  ncols = 2, nrows = 2, hspace=0.5, bottom=0.32)
        fig = self.figures["figureSensitivityBar"]

        grey = Colorful.getDistinctColorList("grey")
        allLabels = []
        for ind,trainingSet in enumerate(self.trainingSetList):
            allLabels = numpy.concatenate((self.sensitivityDataCollection[trainingSet].index.values, allLabels))

        colorList, labelColors = self._getColorsForLabels(allLabels)

        legendLabelColors = {}
        for ind,trainingSet in enumerate(self.trainingSetList):
            ax = fig.getAxes(ind)

            dataframe = self.sensitivityDataCollection[trainingSet].sort_values(by=["MainEffect"], ascending=False)
            dataframe["mathLabel"] = dataframe.apply(lambda row: PlotTweak.getMathLabel(row.name), axis=1)

            nroVariables = dataframe.shape[0]
            margin_bottom = numpy.zeros(nroVariables)

            oneColorList = []

            for variable in dataframe.index:
                indColor = labelColors[variable]
                oneColorList.append(indColor)

                mathlabel = dataframe.loc[variable]["mathLabel"]

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

    def figureLeaveOneOut(self):


        self.figures["figureLeaveOneOut"] = Figure(self.figureFolder,"figureLeaveOneOut",
                                                   figsize = [4.724409448818897, 4],  ncols = 2, nrows = 2,
                                                   bottom = 0.12, hspace = 0.08, wspace=0.04, top=0.95)
        fig = self.figures["figureLeaveOneOut"]

        end = 1.0
        ticks = numpy.arange(0, end + .01, 0.1)
        tickLabels = [f"{t:.1f}" for t in ticks]

        showList = Data.cycleBoolean(len(ticks))

        showList[-1] = False

        for ind,trainingSet in enumerate(self.trainingSetList):
            ax = fig.getAxes(ind)

            dataframe = self.completeDataFrame[trainingSet]

            dataframe = dataframe.loc[dataframe[self.filterIndex]]


            simulated = dataframe[self.responseVariable]
            emulated  = dataframe[self.emulatedVariable]



            stats = self.statsCollection[trainingSet].loc["leaveOneOutStats"]

            slope = stats["slope"]
            intercept = stats["intercept"]
            rSquared = stats["rSquared"]


            dataframe.plot.scatter(ax = ax, x=self.responseVariable, y=self.emulatedVariable,alpha=0.3)




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
                ax.text(-0.25,-0.5, PlotTweak.getUnitLabel("Emulated\ w_{pos}", "m\ s^{-1}"), size=8 , rotation =90)

    def analyseLinearFit(self):
        condition = {}
        updraftVariableName = self.responseVariable

        data = {}
        from scipy import stats

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
        from scipy import stats

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

            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(radiativeWarming, updraft)
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
        from scipy import stats

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


            simulated = dataframe[self.responseVariable]
            emulated  = dataframe[self.emulatedVariable]



            # stats = self.statsCollection[trainingSet].loc["correctedLinearFitStats"]

            # slope = stats["slope"]
            # intercept = stats["intercept"]
            # rSquared = stats["rSquared"]

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

            slopeFit, interceptFit, r_valueFit, p_valueFit, std_errFit = scipy.stats.linregress(simulatedFit, fittedFit)

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


def main():

    figObject = ManuscriptFigures("/home/aholaj/mounttauskansiot/puhtiwork/EmulatorManuscriptData",
                                  "/home/aholaj/Nextcloud/000_WORK/000_ARTIKKELIT/02_LES-Emulator/001_Manuscript_LES_emulator/figures",
                                  "/home/aholaj/mounttauskansiot/puhtiwork/EmulatorManuscriptData/phase02.yaml")

    if True:
        figObject.initReadSensitivityData()
        figObject.figureBarSensitivyData()
    if True:
        figObject.figureLeaveOneOut()
    if True:
        figObject.figureUpdraftLinearFit()
    if True:
        figObject.figureUpdraftCorrectedLinearFit()
    if False:
        figObject.figureUpdraftLinearFitVSEMul()
    if True:
        figObject.figureErrorDistribution()
    if False:
        figObject.figureDistributionOfUpdrafts()
    if False:
        figObject.figureWposVSWposWeighted()

    figObject.finalise()

    # figObject.analyseLinearFitPercentile()
    # figObject.analyseLinearFit()

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("Script completed in " + str(round((end - start),0)) + " seconds")
