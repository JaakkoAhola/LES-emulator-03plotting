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
                            

    def figureUpdraftLinearFit(self):


        self.figures["figureLinearFit"] = Figure(self.figureFolder,"figureLinearFit", figsize = [4.724409448818897, 4],
                                                 ncols = 2, nrows = 2, bottom = 0.11, hspace = 0.08, wspace=0.12, top=0.94)
        fig = self.figures["figureLinearFit"]

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

            updraftVariableName = self.responseVariable
            
            condition["notMatchObservation"] = ~ ( (dataframe[updraftVariableName] > dataframe["drflx"]*self.observationParameters["slope"]+ self.observationParameters["intercept"]-self.observationParameters["error"]) &\
                                                   (dataframe[updraftVariableName] < dataframe["drflx"]*self.observationParameters["slope"]+ self.observationParameters["intercept"]+self.observationParameters["error"]))

            dataFrameInside = dataframe[ ~condition["notMatchObservation"]]
            percentageInside = len(dataFrameInside) / len(dataframe) *100.

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
                                    xPosition=PlotTweak.getXPosition(ax, 0.02), yPosition = PlotTweak.getYPosition(ax, 0.93))

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

            ax.text(-25, 0.67,
                f"""{PlotTweak.getLatexLabel('y=a + b * x')}
{PlotTweak.getLatexLabel(f'a={intercept:.4f}')}
{PlotTweak.getLatexLabel(f'b={slope:.6f}')}
{PlotTweak.getLatexLabel(f'R^2={rSquared:.2f}')}
{PlotTweak.getLatexLabel('p_{in}=' + f'{percentageInside:.1f}')}%""", fontsize = 6)

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
                




    def tables_featureImportanceOrder(self):
        self.labelCategorised.to_latex(self.figureFolder / "featureImportanceOrder.tex", columns =["mathLabel", "relativeCombined", "zeros"], index =False, float_format="{:.2e}")

def main():

    figObject = ManuscriptFigures("/home/aholaj/mounttauskansiot/puhtiwork/EmulatorManuscriptData",
                                  "/home/aholaj/Nextcloud/000_WORK/000_ARTIKKELIT/02_LES-Emulator/001_Manuscript_LES_emulator/figures",
                                  "/home/aholaj/mounttauskansiot/puhtiwork/EmulatorManuscriptData/phase02.yaml")

    if False:
        figObject.initReadFeatureImportanceData()
        figObject.figureBarFeatureImportanceData()
        figObject.tables_featureImportanceOrder()
    if True:
        figObject.figureUpdraftLinearFit()
    if False:
        figObject.figureUpdraftCorrectedLinearFit()
    if False:
        figObject.figureUpdraftLinearFitVSEMul()

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
