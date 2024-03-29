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
                self.filteredSourceData = pandas.read_csv( localPath / "eclair_dataset_2001_designvariables.csv", index_col = 0 )
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
        ncols = len(self.featureImportanceDataCollection[list(self.featureImportanceDataCollection)[0]]) # number of methods
        nrows = len(self.featureImportanceDataCollection) # = number of training sets

        self.figures["figureFeatureImportanceBar"] = Figure(self.figureFolder,"figure6",
                                                            figsize=[self.figureWidth,6],
                                                            ncols = ncols, nrows = nrows,
                                                            hspace=0.8, bottom=0.20, wspace = 0.05, top = 0.97)
        fig = self.figures["figureFeatureImportanceBar"]

        maksimi = 0
        column = "relativeImportance"
        for row,trainingSet in enumerate(list(self.featureImportanceDataCollection)):
            if self.featureImportanceDataCollection[trainingSet] is None:
                continue
            for col, predictor in enumerate(list(self.featureImportanceDataCollection[trainingSet])[::-1]):

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
                            PlotTweak.getLatexLabel(self.trainingSetSensibleNames[ind//ncols]), size=8 , rotation =90)
            if ind == 1:
                ax.text(PlotTweak.getXPosition(ax, 0.2), PlotTweak.getYPosition(ax, 1.05), self.predictorClearNames["emulator"], size=8)
            if ind == 0:
                ax.text(PlotTweak.getXPosition(ax, 0.2),PlotTweak.getYPosition(ax, 1.05), self.predictorClearNames["correctedLinearFit"], size=8)

        fig.getAxes(-1).legend(handles=PlotTweak.getPatches(self.mathLabelColors),
                      loc=(-0.9,-1.59),
                      ncol = self.legendCols,
                      fontsize = 8)



    def figurePredictorsVsSimulated(self):
        numberOfMethods = 3
        self.figures["figurePredictorsVsSimulated"] = Figure(self.figureFolder,"figure5",
                                                   figsize = [self.figureWidth, 7],  ncols = numberOfMethods, nrows = 4,
                                                   bottom = 0.07, hspace = 0.09, wspace=0.10, top=0.95, left=0.16, right = 0.98)
        fig = self.figures["figurePredictorsVsSimulated"]

        start = 0.0
        end = 1.0
        ticks = numpy.arange(0, end + .01, 0.1)
        tickLabels = [f"{t:.1f}" for t in ticks]

        showList = Data.cycleBoolean(len(ticks))

        showList[0] = False
        showList[-1] = False

        for row,trainingSet in enumerate(self.trainingSetList):
            for col, predictor in enumerate(self.predictionVariableList):
                ax = fig.getAxesGridPoint( {"row": row, "col": col})
                shortname = self.predictorShortNames[col]
                dataframe = self.completeDataFrame[trainingSet]
                if self.completeDataFrame[trainingSet] is None or self.statsCollection[trainingSet] is None:
                    continue

                dataframe = dataframe.loc[dataframe[self.filterIndex]]

                simulated = dataframe[self.responseVariable]

                statistics = self.statsCollection[trainingSet].loc[ self.predictorShortNames[col] ]

                slope = statistics["slope"]
                intercept = statistics["intercept"]
                rSquared = statistics["rSquared"]
                rmse = statistics["rmse"]


                dataframe.plot.scatter(ax = ax, x=self.responseVariable, y=predictor,color = self.predictorColors[ shortname ], alpha=0.3)

                coef = [slope, intercept]
                poly1d_fn = numpy.poly1d(coef)
                ax.plot(simulated.values, poly1d_fn(simulated.values), color = "k")

                ax.set_ylim([start, end])

                ax.set_xlim([start, end])


                PlotTweak.setAnnotation(ax, f"""{PlotTweak.getLatexLabel(f'R^2={rSquared:.2f}','')}
{PlotTweak.getLatexLabel(f'RMSE={rmse:.4f}','')}""",
                                        xPosition=0.23, yPosition=0.05, bbox_props = None)

                PlotTweak.setXaxisLabel(ax,"")
                PlotTweak.setYaxisLabel(ax,"")




                ax.set_xticks(ticks)
                ax.set_xticklabels(tickLabels)
                PlotTweak.hideLabels(ax.xaxis, showList)

                ax.set_yticks(ticks)
                ax.set_yticklabels(tickLabels)
                PlotTweak.hideLabels(ax.yaxis, showList)

                PlotTweak.setXTickSizes(ax, Data.cycleBoolean(len(ticks)))
                PlotTweak.setYTickSizes(ax, Data.cycleBoolean(len(ticks)))


        for ind in range(12):
            ax = fig.getAxes(ind)

            PlotTweak.setAnnotation(ax, f"({Data.getNthLetter(ind)})",
                                        xPosition=ax.get_xlim()[1]*0.05, yPosition = ax.get_ylim()[1]*0.90)

            if ind not in numpy.asarray(range(4))*3:
                PlotTweak.hideYTickLabels(ax)
            else:
                ax.text(PlotTweak.getXPosition(ax, -0.56), PlotTweak.getYPosition(ax, 0.3),
                        PlotTweak.getLatexLabel(self.trainingSetSensibleNames[ind//3]), size=8 , rotation =90)

            if ind not in list(range(9,12)):
                PlotTweak.hideXTickLabels(ax)

            if ind == 1:
                collectionOfLabelsColors = dict(zip(list(self.predictorClearNames.values()), list(self.predictorColors.values())))
                legendLabelColors = PlotTweak.getPatches(collectionOfLabelsColors)

                artist = ax.legend( handles=legendLabelColors, loc=(-0.4, 1.05), frameon = True, framealpha = 1.0, ncol = 3 )

                ax.add_artist(artist)

            if ind == 3:
                ax.text(PlotTweak.getXPosition(ax, -0.42), PlotTweak.getYPosition(ax, -0.5),
                        PlotTweak.getUnitLabel("Predicted\ w_{pos}", "m\ s^{-1}"), size=8 , rotation =90)
            if ind == 10:
                ax.text(-0.1,-0.27,PlotTweak.getUnitLabel("Simulated\ w_{pos}", "m\ s^{-1}") , size=8)

    def figureSimulatedUpdraft_vs_CloudRadiativeCooling(self):
        self.__figureUpdraft_vs_CloudRadiativeCooling(self.responseVariable, {"fig" : "figure4", "legend" : "Simulated"}, Colorful.getDistinctColorList("orange"))
    def figureLinearFitUpdraft_vs_CloudRadiativeCooling(self):
        self.__figureUpdraft_vs_CloudRadiativeCooling(self.linearFitVariable, {"fig" : "figureLinearFitUpdraft_vs_CloudRadiativeCooling", "legend" : self.predictorClearNames["linearFit"]}, Colorful.getDistinctColorList("red"))

    def figureCorrectedLinearUpdraft_vs_CloudRadiativeCooling(self):
        self.__figureUpdraft_vs_CloudRadiativeCooling(self.correctedLinearFitVariable, {"fig" : "figureCorrectedLinearUpdraft_vs_CloudRadiativeCooling", "legend" :self.predictorClearNames["correctedLinearFit"]}, Colorful.getDistinctColorList("green"))

    def __figureUpdraft_vs_CloudRadiativeCooling(self, updraftVariableName, names : dict, dataColor):


        self.figures[names["fig"]] = Figure(self.figureFolder,names["fig"], figsize = [self.figureWidth, 4],
                                                 ncols = 2, nrows = 2, bottom = 0.11, hspace = 0.08, wspace=0.12, top=0.94)
        fig = self.figures[names["fig"]]

        xstart = -140
        xend = 50

        ystart = 0.0
        yend = 1.0
        yticks = numpy.arange(0, yend + .01, 0.1)
        tickLabels = [f"{t:.1f}" for t in yticks]

        yShowList = Data.cycleBoolean(len(yticks))

        color_obs = Colorful.getDistinctColorList("grey")
        condition = {}
        for ind,trainingSet in enumerate(self.trainingSetList):
            ax = fig.getAxes(ind)

            dataframe = self.completeDataFrameFiltered[trainingSet]

            if dataframe is None:
                continue

            condition["notMatchObservation"] = ~ ( (dataframe[updraftVariableName] > dataframe["drflx"]*self.observationParameters["slope"]+ self.observationParameters["intercept"]-self.observationParameters["error"]) &\
                                                   (dataframe[updraftVariableName] < dataframe["drflx"]*self.observationParameters["slope"]+ self.observationParameters["intercept"]+self.observationParameters["error"]))

            dataFrameInside = dataframe[ ~condition["notMatchObservation"]]
            percentageInside = len(dataFrameInside) / len(dataframe) *100.

            radiativeCooling  = dataframe["drflx"].values
            updraft =  dataframe[updraftVariableName].values


            poly1d_Observation = numpy.poly1d(numpy.asarray([self.observationParameters["slope"],self.observationParameters["intercept"] ])) #?0.44 ×CTRC+
            ax.plot(radiativeCooling, poly1d_Observation(radiativeCooling), color = color_obs)
            ax.fill_between(sorted(radiativeCooling),
                            poly1d_Observation(sorted(radiativeCooling)) - self.observationParameters["error"]*numpy.ones(numpy.shape(radiativeCooling)), poly1d_Observation(sorted(radiativeCooling)) + self.observationParameters["error"]*numpy.ones(numpy.shape(radiativeCooling)),
                            alpha=0.2)

            slope, intercept, r_value, p_value, std_err = stats.linregress(radiativeCooling, updraft)
            coef = [slope, intercept]
            rSquared = numpy.power(r_value, 2)

            fitColor = "k"
            dataframe.plot.scatter(ax = ax, x="drflx", y=updraftVariableName, alpha=0.3, color = dataColor)

            poly1d_fn = numpy.poly1d(coef)

            linearFit = []
            for radCoolingValue in list(self.completeDataFrame[trainingSet]["drflx"]):
                linearFit.append(poly1d_fn(radCoolingValue))

            self.completeDataFrame[trainingSet][self.linearFitVariable] = linearFit
            ax.plot(radiativeCooling, poly1d_fn(radiativeCooling), color = fitColor)

            ax.set_xlim([xstart, xend])
            ax.set_ylim([ystart, yend])


            PlotTweak.setAnnotation(ax, self.annotationCollection[trainingSet],
                                    xPosition=PlotTweak.getXPosition(ax, 0.02), yPosition = PlotTweak.getYPosition(ax, 0.93))

            PlotTweak.setXaxisLabel(ax,"")
            PlotTweak.setYaxisLabel(ax,"")


            xticks = PlotTweak.setXticks(ax, start= xstart, end = xend, interval = 10, integer=True)

            xShownLabelsBoolean = PlotTweak.setXLabels(ax, xticks, start= xstart, end = xend, interval = 40)
            xShownLabelsBoolean = Data.cycleBoolean(len(xShownLabelsBoolean))
            PlotTweak.setXTickSizes(ax, xShownLabelsBoolean)


            if ind == 0:
                collectionOfLabelsColors = {names["legend"] : dataColor, "Fit" : "k", "Zheng et al. 2016" : color_obs}
                legendLabelColors = PlotTweak.getPatches(collectionOfLabelsColors)

                artist = ax.legend( handles=legendLabelColors, loc=(0.17, 1.02), frameon = True, framealpha = 1.0, ncol = 3 )

                ax.add_artist(artist)

            ax.text(-25, 0.67,
                f"""{PlotTweak.getLatexLabel('y=a + b * x')}
{PlotTweak.getLatexLabel(f'a={intercept:.4f}')}
{PlotTweak.getLatexLabel(f'b={slope:.6f}')}
{PlotTweak.getLatexLabel(f'R^2={rSquared:.2f}')}""", fontsize = 6)

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
                        PlotTweak.getUnitLabel(names["legend"] + "\ w_{pos}", "m\ s^{-1}"), size=8 , rotation =90)
            if ind == 2:
                ax.text(-50,-0.25, PlotTweak.getUnitLabel("Cloud\ top\ radiative\ cooling", "W\ m^{-2}"), size=8)




    def figureDesignVariables(self):
        nrows = 4
        ncols = ceil(len(self.designVariablePool) / nrows)

        self.figures["figureDesignVariables"] = Figure(self.figureFolder,"figure2",
                                                   figsize = [self.figureWidth, 7],  ncols = ncols, nrows = nrows,
                                                   bottom = 0.04, hspace = 0.17, wspace=0.07, top=0.98, left=0.05, right = 0.99)
        fig = self.figures["figureDesignVariables"]
        rightUp = [0.57, 0.70]
        leftUp = [0.25, 0.73]
        leftDown = [0.1, 0]
        rightDown = [0.45, 0.05]
        middleDown = [0.33, 0.05]
        default = [0.5,0.5]
        specsPositions = [ [0.3, 0.05], [0.2, 0.05], [0.3,0.5],
                        [0.3,0.5], [0.3, 0.4], [0.3,0.5],
                        [0.3,0.5], [0.3,0.5], [0.3,0.5],
                        [0.2, 0.05], middleDown
                        ]

        aeroNumberVariables = ["ks", "as", "cs"]
        reff = "rdry_AS_eff"

        minisDesigns = [numpy.nan]*len(self.designVariablePool)
        maxisDesigns = [numpy.nan]*len(self.designVariablePool)

        for ind,variable in enumerate(self.designVariablePool):
            ax = fig.getAxes(ind)

            minimi = numpy.nan
            maximi = numpy.nan
            if variable == "pblh":
                variableSourceName = "pbl"
            else:
                variableSourceName = variable

            if hasattr(self, "filteredSourceData") and (self.filteredSourceData is not None):
                sourceDataVariable = self.filteredSourceData[variableSourceName]
                sourceDataVariable = Data.dropInfNanFromDataFrame(sourceDataVariable)
                if variable in aeroNumberVariables:
                    sourceDataVariable = sourceDataVariable*1e-6
                if variable == reff:
                    sourceDataVariable = sourceDataVariable*1e9
                sourceDataVariable = Data.dropInfNanFromDataFrame(sourceDataVariable)

                if variable == "cos_mu":
                    sourceDataVariable = sourceDataVariable[sourceDataVariable > Data.getEpsilon()]
                sourceDataVariable.plot.density(ax = ax, color =Colorful.getDistinctColorList("grey"))




                if variable in (aeroNumberVariables + [reff , "cdnc"]):
                    print(f"{variable} ECHAM mean {sourceDataVariable.mean():.2f}, ECHAM median {sourceDataVariable.median():.2f}")


            for tt, trainingSet in enumerate(self.trainingSetList):

                if variable in self.completeDataFrame[trainingSet].keys():

                    if self.completeDataFrame[trainingSet] is None:
                        continue

                    trainingSetVariable = self.completeDataFrame[trainingSet][variable]
                    if variable in aeroNumberVariables:
                        trainingSetVariable = trainingSetVariable*1e-6
                    if variable == reff:
                        trainingSetVariable = trainingSetVariable*1e9

                    trainingSetVariable = Data.dropInfNanFromDataFrame(trainingSetVariable)



                    minimi = numpy.nanmin([minimi, trainingSetVariable.min()])
                    maximi = numpy.nanmax( [maximi, trainingSetVariable.max()])

                    minisDesigns[ind] = minimi
                    maxisDesigns[ind] = maximi
                    trainingSetVariable.plot.density(ax = ax, color = self.trainingSetColors[trainingSet])

                    if variable in (aeroNumberVariables + [reff , "cdnc"]):
                        print(f"{variable} {trainingSet} mean {trainingSetVariable.mean():.2f} {trainingSet} median {trainingSetVariable.median():.2f}")


            variableAnnotationYposition = 0.9

            if variable == "cos_mu":
                annotationOfVariable = PlotTweak.getMathLabel(variable)

            else:
                annotationOfVariable = PlotTweak.getUnitLabel(PlotTweak.getMathLabelFromDict(variable), PlotTweak.getVariableUnit(variable))

            annotation = f"({Data.getNthLetter(ind)}) {annotationOfVariable}"
            PlotTweak.setAnnotation(ax, annotation,
                                    xPosition =  0.2,
                                    yPosition = variableAnnotationYposition,
                                    xycoords = "axes fraction")
            ax.set_ylabel("")

            ax.set_xlim([minimi, maximi])
            ax.set_ylim([0,ax.get_ylim()[1]])


            # fixing yticks with matplotlib.ticker "FixedLocator"
            label_format = '{:,.0f}'
            ticks_loc = ax.get_yticks().tolist()
            ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
            yticklabels =[label_format.format(x) for x in ticks_loc]
            yticklabels[0] = "0"
            ax.set_yticklabels(yticklabels)


            if ind in [0,3,6,9,12]:
                matplotlib.pyplot.setp(ax.get_yticklabels()[0], visible=True)
                matplotlib.pyplot.setp(ax.get_yticklabels()[1:], visible=False)
            else:
                PlotTweak.hideYTickLabels(ax)



        ax = fig.getAxes(11)
        ax.axis("off")

        legendLabelColors = PlotTweak.getPatches(self.allSetColors)

        artist = ax.legend( handles=legendLabelColors, loc=(0.0, 0.00), frameon = True, framealpha = 1.0, ncol = 1 )

        ax.add_artist(artist)

        for ind,variable in enumerate(self.designVariablePool):
            ax = fig.getAxes(ind)


            variableSpecs = f"""{PlotTweak.getLatexLabel(f'min={minisDesigns[ind]:.2f}')}
{PlotTweak.getLatexLabel(f'max={maxisDesigns[ind]:.2f}')}"""

            ax.annotate( variableSpecs, xy=specsPositions[ind], size=8, bbox = dict(pad = 0.6, fc="w", ec="w", alpha=0.9), xycoords = "axes fraction")

            if ind in [1,2,3]:
                ax.set_xlim([0,maxisDesigns[ind]])
            if ind in [5, 6,7,8]:
                limits = {5:200,  6:1000, 7:300, 8:15}
                ax.set_xlim([0,limits[ind]])
                matplotlib.pyplot.setp(ax.get_xticklabels()[-1], visible=False)
            if ind == 10:
                ax.set_xlim([0,1])
                label_format = '{:,.2f}'
                ticks_loc = ax.get_xticks().tolist()
                ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
                xticklabels =[label_format.format(x) for x in ticks_loc]
                xticklabels[0] = "0"
                xticklabels[-1] = "1"
                ax.set_xticklabels(xticklabels)


    def table_featureImportanceStats(self):
        for trainingSet in self.featureImportanceDataCollection:
            if self.featureImportanceDataCollection[trainingSet] is None:
                continue
            for predictor in self.featureImportanceDataCollection[trainingSet]:

                dataframe = self.featureImportanceDataCollection[trainingSet][predictor]
                clearName = self.predictorClearNames[ self.predictorShortNameDict[predictor] ]
                clearName = clearName.replace(" ", "#")
                dataframe.rename(columns = {"mathLabel":clearName}, inplace = True)
                self.__latexTableWrapper(dataframe,
                                         f"table_featureImportanceStats_{trainingSet}_{predictor.split('_')[-1]}",
                                         [clearName, "Mean", "Std", "relativeImportance"],
                                         float_format="{:.3f}".format)

    def tables_featureImportanceOrder(self):
        self.__latexTableWrapper(self.labelCategorised,
                                 "tables_featureImportanceOrder",
                                 ["mathLabel", "relativeMean", "LFRF relative mean", "GPE relative mean" , "zeros"],
                                 float_format="{:.3f}".format, column_format = "| m{1.2cm}  |m{2cm}  | m{1cm} | m{2cm} |m{2cm} |")

    def tables_predictorsVsSimulated(self):
        for trainingSet in self.trainingSetList:
            if self.statsCollection[trainingSet] is None:
                continue
            statistics = self.statsCollection[trainingSet].loc[ self.predictorShortNames ]

            firstColumnName = self.trainingSetSensibleDict[trainingSet].replace("\ ", "#")
            statistics[firstColumnName] = self.predictorClearNames

            self.__latexTableWrapper(statistics,
                                     f"tables_predictorsVsSimulated_{trainingSet}",
                                     [firstColumnName, "rSquared","r_value", "rmse"],
                                     float_format="{:.3f}".format)

    def tables_bootstraps(self):
        for trainingSet in self.trainingSetList:
            if self.bootstrapCollection[trainingSet] is None:
                continue
            self.bootstrapCollection[trainingSet].insert(loc = 0, column = self.trainingSetSensibleDict[trainingSet].replace("\ ", "#"), value = self.bootstrapCollection[trainingSet].index.values)
            self.__latexTableWrapper( self.bootstrapCollection[trainingSet], f"Bootstrap_{trainingSet}", columns = None, float_format = "{:.3f}".format, index = False)

    def __latexTableWrapper(self, table, fileName, columns, float_format = "{:.2e}".format, index = False, column_format = None):
        table.to_latex(self.tableFolder / (fileName + ".tex"), columns = columns, index = index, float_format=float_format, column_format = column_format)

    def finaliseLatexTables(self):
        for texFile in self.tableFolder.glob("**/*.tex"):
            linesToBeWritten = []
            with open(texFile, "r") as readFile:
                for line in readFile:

                    line = line.replace(" ", "")

                    line = line.replace("toprule", "tophline")

                    line = line.replace("midrule", "middlehline")
                    line = line.replace("bottomrule", "bottomhline")

                    line = line.replace("\\$\\textbackslashmathbf\\{\\{cos\\_\\{\\textbackslashmu\\}\\}\\{\\textbackslash\\}\\}\\$", PlotTweak.getMathLabelTableFormat("cos_mu"))
                    line = line.replace("\\$\\textbackslashmathbf\\{\\{N\\_\\{" +  PlotTweak.getMathLabelSubscript("as") +"\\}\\}\\{\\textbackslash\\}\\}\\$", PlotTweak.getMathLabelTableFormat("as"))
                    line = line.replace("\\$\\textbackslashmathbf\\{\\{N\\_\\{" + PlotTweak.getMathLabelSubscript("ks") +"\\}\\}\\{\\textbackslash\\}\\}\\$", PlotTweak.getMathLabelTableFormat("ks"))
                    line = line.replace("\\$\\textbackslashmathbf\\{\\{N\\_\\{" + PlotTweak.getMathLabelSubscript("cs") +"\\}\\}\\{\\textbackslash\\}\\}\\$", PlotTweak.getMathLabelTableFormat("cs"))
                    line = line.replace("\\$\\textbackslashmathbf\\{\\{w\\_\\{lin.fit\\}\\}\\{\\textbackslash\\}\\}\\$", PlotTweak.getMathLabelTableFormat("w2pos_linearFit"))
                    line = line.replace("\\$\\textbackslashmathbf\\{\\{\\{\\textbackslashtheta\\}\\_\\{L\\}\\}\\{\\textbackslash\\}\\}\\$", PlotTweak.getMathLabelTableFormat("tpot_pbl"))
                    line = line.replace("\\$\\textbackslashmathbf\\{\\{\\textbackslashDelta\\{\\textbackslashtheta\\}\\_\\{L\\}\\}\\{\\textbackslash\\}\\}\\$", PlotTweak.getMathLabelTableFormat("tpot_inv"))
                    line = line.replace("\\$\\textbackslashmathbf\\{\\{\\textbackslashDeltaq\\_\\{L\\}\\}\\{\\textbackslash\\}\\}\\$", PlotTweak.getMathLabelTableFormat("q_inv"))
                    line = line.replace("\\$\\textbackslashmathbf\\{\\{r\\_\\{eff\\}\\}\\{\\textbackslash\\}\\}\\$", PlotTweak.getMathLabelTableFormat("rdry_AS_eff"))
                    line = line.replace("\\$\\textbackslashmathbf\\{\\{H\\_\\{PBL\\}\\}\\{\\textbackslash\\}\\}\\$", PlotTweak.getMathLabelTableFormat("pblh"))


                    line = line.replace("mathLabel", "Variable")
                    line = line.replace("relativeCombined", "Product of relative permutation feature importances")
                    line = line.replace("relativeMean", "Mean of relative permutation feature importances")
                    line = line.replace("zeros", "Number of times relative importance equal to zero")
                    line = line.replace("relativeImportance", "Relative permutation feature importance")
                    line = line.replace("LFRFrelativemean", "LFRF relative mean")
                    line = line.replace("GPErelativemean", "GPE relative mean")

                    line = line.replace("rSquared", "$R^2$")
                    line = line.replace("r\\_value", "R")

                    line = line.replace("emulator", self.predictorClearNames["emulator"])
                    line = line.replace("linearFit", self.predictorClearNames["linearFit"])
                    line = line.replace("correctedLinearFit", self.predictorClearNames["correctedLinearFit"])

                    line = line.replace("\\_Mean", " mean")
                    line = line.replace("\\_Std", " std")
                    line = line.replace("\\#", " ")

                    linesToBeWritten.append(line)
            texFile.unlink()
            with open(texFile, "w") as writeFile:
                for line in linesToBeWritten:
                    writeFile.write(line)


def main():
    try:
        locationsFile = sys.argv[1]
    except IndexError:
        locationsFile = "/home/aholaj/mounttauskansiot/puhtiwork/EmulatorManuscriptData_3/locationsMounted.yaml"

    figObject = ManuscriptFigures(locationsFile)

    printTables = False
    if True:
        #figure6
        figObject.initReadFeatureImportanceData()
        figObject.figureBarFeatureImportanceData()
        if printTables: figObject.tables_featureImportanceOrder()
        if printTables: figObject.table_featureImportanceStats()
    if True:
        #figure4
        figObject.figureSimulatedUpdraft_vs_CloudRadiativeCooling()


    if True:
        #figure5
        figObject.figurePredictorsVsSimulated()
        if printTables: figObject.tables_predictorsVsSimulated()

    if True:
        if printTables: figObject.tables_bootstraps()

    if True:
        #figure2
        figObject.initReadFilteredSourceData()
        figObject.figureDesignVariables()


    figObject.finalise()
    if printTables: figObject.finaliseLatexTables()

if __name__ == "__main__":
    start = time.time()
    now = datetime.now().strftime('%d.%m.%Y %H.%M')
    print(f"Script started {now}.")
    main()
    end = time.time()
    now = datetime.now().strftime('%d.%m.%Y %H.%M')
    print(f"Script completed {now} in {Data.timeDuration(end - start)}")
