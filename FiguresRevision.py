#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 13.12.2021

@author: Jaakko Ahola, Finnish Meteorological Institute
@licence: MIT licence Copyright
"""
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
from copy import deepcopy

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
from plotEMUL import ManuscriptFigures

class FiguresRevision(ManuscriptFigures):

    def figure_relative_relative_change_lwp(self):
        name = "figure_relative_relative_change_lwp"
        self.figures[name] = Figure(self.figureFolder,name, figsize = [self.figureWidth, 4],
                                    ncols=2, nrows=2,
                                    hspace=0.08, wspace=0.12,
                                    bottom=0.11, top=0.93,
                                    )

        fig = self.figures[name]
        blue_color = Colorful.getDistinctColorList("blue")
        # red_color = Colorful.getDistinctColorList("red")
        for ind, trainingSet in enumerate(self.trainingSetList):
            current_axes = fig.getAxes(ind)
            # dataframe = self.completeDataFrame[trainingSet]
            dataframe_filtered = self.completeDataFrameFiltered[trainingSet]

            # dataframe["lwp_rwp_relative_change"].plot.hist(ax=current_axes, color = red_color )
            dataframe_filtered["lwp_rwp_relative_change"].plot.hist(ax=current_axes, color = blue_color)



        # tweaks
        start = 0.0
        end = 2.0
        ticks = numpy.arange(0, end + .01, 0.5)
        tickLabels = [f"{t:.1f}" for t in ticks]

        showList = Data.cycleBoolean(len(ticks))

        showList[0] = True
        showList[-1] = False
        for ind, trainingSet in enumerate(self.trainingSetList):
            current_axes = fig.getAxes(ind)

            if "4" in trainingSet:
                yticks = numpy.arange(0, 151, 25)
                ytickLabels = [f"{t:d}" for t in yticks]

            else:
                yticks = numpy.arange(0, 501, 100)
                ytickLabels = [f"{t:d}" for t in yticks]

            yshowList = Data.cycleBoolean(len(yticks))
            yshowList[-1] =False

            current_axes.set_xlim([start, end])
            PlotTweak.setYaxisLabel(current_axes,"")



            if ind in [0,1]:
                PlotTweak.hideXTickLabels(current_axes)
            if ind in [1,3]:
                PlotTweak.hideYTickLabels(current_axes)

            current_axes.set_xticks(ticks)
            current_axes.set_xticklabels(tickLabels)
            PlotTweak.hideLabels(current_axes.xaxis, showList)

            current_axes.set_yticks(yticks)
            current_axes.set_yticklabels(ytickLabels)
            PlotTweak.hideLabels(current_axes.yaxis, yshowList)

            PlotTweak.setAnnotation(current_axes,
                                    self.annotationCollection[trainingSet],
                                    xPosition=PlotTweak.getXPosition(current_axes, 0.05),
                                    yPosition = PlotTweak.getYPosition(current_axes, 0.9))

            if ind == 0:
                collectionOfLabelsColors = {"Total water path (LWP + RWP) relative change" : blue_color,
                                            #"LWP relative change (all LES data)" : red_color,
                                            }
                legendLabelColors = PlotTweak.getPatches(collectionOfLabelsColors)

                artist = current_axes.legend(handles=legendLabelColors,
                                             loc=(0.0, 1.03),
                                             frameon=True,
                                             framealpha=1.0,
                                             ncol=1)

                current_axes.add_artist(artist)

            if ind == 0:
                current_axes.text(PlotTweak.getXPosition(current_axes, -0.25), PlotTweak.getYPosition(current_axes, -0.25),
                                  "Frequency", size=8, rotation=90)

    def figure_lwp_scatter_plot(self):
        name = "figure_lwp_scatter_plot"
        self.figures[name] = Figure(self.figureFolder,name, figsize = [self.figureWidth, 4],
                                    ncols=2, nrows=2,
                                    hspace=0.08, wspace=0.12,
                                    bottom=0.11, top=0.93,
                                    )

        fig = self.figures[name]
        blue_color = Colorful.getDistinctColorList("blue")
        # red_color = Colorful.getDistinctColorList("red")
        for ind, trainingSet in enumerate(self.trainingSetList):
            current_axes = fig.getAxes(ind)

            dataframe_filtered = deepcopy(self.completeDataFrameFiltered[trainingSet])
            dataframe_filtered["total_wp"] = dataframe_filtered.apply(lambda row:\
                                                    (row["rwp_last_hour"] + row["lwp_last_hour"])*1e3,
                                                    axis=1)

            print("lwp", trainingSet, dataframe_filtered["total_wp"].max(), dataframe_filtered["lwp"].max())

            dataframe_filtered.plot.scatter(x="lwp",
                                            y="total_wp",
                                            ax=current_axes,
                                            color = blue_color,
                                            alpha=0.3)

            current_axes.axline([0, 0], [1, 1], color="k")

        # tweaks
        start = 0.0
        end = 800
        interval = 100
        ticks = numpy.arange(0, end + .01, interval)
        tickLabels = [f"{t:.0f}" for t in ticks]

        showList = Data.cycleBoolean(len(ticks))

        showList[0] = True
        showList[-1] = False
        for ind, trainingSet in enumerate(self.trainingSetList):
            current_axes = fig.getAxes(ind)

            current_axes.set_xlim([start, end])
            PlotTweak.setYaxisLabel(current_axes,"")
            PlotTweak.setXaxisLabel(current_axes,"")



            if ind in [0,1]:
                PlotTweak.hideXTickLabels(current_axes)
            if ind in [1,3]:
                PlotTweak.hideYTickLabels(current_axes)

            current_axes.set_xticks(ticks)
            current_axes.set_xticklabels(tickLabels)
            PlotTweak.hideLabels(current_axes.xaxis, showList)

            current_axes.set_yticks(ticks)
            current_axes.set_yticklabels(tickLabels)
            PlotTweak.hideLabels(current_axes.yaxis, showList)

            PlotTweak.setAnnotation(current_axes,
                                    self.annotationCollection[trainingSet],
                                    xPosition=PlotTweak.getXPosition(current_axes, 0.05),
                                    yPosition = PlotTweak.getYPosition(current_axes, 0.9))

            if ind == 0:
                collectionOfLabelsColors = {"Total water path (LWP + RWP) change" : blue_color,
                                            #"LWP relative change (all LES data)" : red_color,
                                            }
                legendLabelColors = PlotTweak.getPatches(collectionOfLabelsColors)

                artist = current_axes.legend(handles=legendLabelColors,
                                             loc=(0.0, 1.03),
                                             frameon=True,
                                             framealpha=1.0,
                                             ncol=1)

                current_axes.add_artist(artist)

            if ind == 3:
                current_axes.text(PlotTweak.getXPosition(current_axes, -0.7), PlotTweak.getYPosition(current_axes, -0.25),
                                   PlotTweak.getUnitLabel("Total\ water\ path\ in\ the\ beginning", PlotTweak.getVariableUnit("lwp")), size=8)
            if ind == 0:
                current_axes.text(PlotTweak.getXPosition(current_axes, -0.27), PlotTweak.getYPosition(current_axes, -0.8),
                                  PlotTweak.getUnitLabel("Total\ water\ path\ average\ of\ last\ hour", PlotTweak.getVariableUnit("lwp")), size=8, rotation=90)


    def figure_surface_precipitation_accumulated(self):
        name = "figure_surface_precipitation_accumulated"
        self.figures[name] = Figure(self.figureFolder,name, figsize = [self.figureWidth, 4],
                                                 ncols=2, nrows=2,
                                                 bottom=0.11, top=0.87,
                                                 hspace=0.18, wspace=0.24,
                                                 right=0.95
                                                 )
        fig = self.figures[name]
        blue_color = Colorful.getDistinctColorList("blue")
        red_color = Colorful.getDistinctColorList("red")

        for ind, trainingSet in enumerate(self.trainingSetList):
            current_axes = fig.getAxes(ind)
            dataframe = self.completeDataFrame[trainingSet]
            dataframe_filtered = self.completeDataFrameFiltered[trainingSet]
            print(trainingSet, numpy.log10(dataframe["surface_precipitation_accumulated"].max()))
            dataframe["surface_precipitation_accumulated"].plot.hist(ax=current_axes, color = red_color )
            dataframe_filtered["surface_precipitation_accumulated"].plot.hist(ax=current_axes, color = blue_color)

        # tweaks


        for ind, trainingSet in enumerate(self.trainingSetList):
            if "4" in trainingSet:
                start = 0.0
                end = 4e-4
                interval = 1e-4
                ticks = numpy.arange(0, end + interval, interval)
                tickLabels = [f"{t:.0e}" for t in ticks]
                tickLabels[0] = "0"

                yticks = numpy.arange(0, 151, 25)
                ytickLabels = [f"{t:d}" for t in yticks]

            else:
                start = 0.0
                end = 4e-3
                interval = 1e-3
                ticks = numpy.arange(0, end + interval, interval)
                tickLabels = [f"{t:.0e}" for t in ticks]
                tickLabels[0] = "0"

                yticks = numpy.arange(0, 501, 100)
                ytickLabels = [f"{t:d}" for t in yticks]


            showList = Data.cycleBoolean(len(ticks))

            showList[0] = True
            showList[-1] = True

            yshowList = Data.cycleBoolean(len(yticks))
            yshowList[-1] =False

            current_axes = fig.getAxes(ind)

            current_axes.set_xlim([start, end])

            PlotTweak.setYaxisLabel(current_axes,"")
            current_axes.set_xticks(ticks)
            current_axes.set_xticklabels(tickLabels)
            PlotTweak.hideLabels(current_axes.xaxis, showList)

            current_axes.set_yticks(yticks)
            current_axes.set_yticklabels(ytickLabels)
            PlotTweak.hideLabels(current_axes.yaxis, yshowList)

            if ind == 0:
                collectionOfLabelsColors = {"Accumulated surface precipitation (LES filtered data)" : blue_color,
                                            "Accumulated surface precipitation (all LES data)" : red_color,}
                legendLabelColors = PlotTweak.getPatches(collectionOfLabelsColors)

                artist = current_axes.legend(handles=legendLabelColors,
                                             loc=(0.0, 1.03),
                                             frameon=True,
                                             framealpha=1.0,
                                             ncol=1)

                current_axes.add_artist(artist)

            if ind == 3:
                current_axes.text(PlotTweak.getXPosition(current_axes, -0.3), PlotTweak.getYPosition(current_axes, -0.25),
                                  "(" + PlotTweak.getLatexLabel("kg\ m^{-2} s^{-1}") + ")", size=8)
            if ind == 0:
                current_axes.text(PlotTweak.getXPosition(current_axes, -0.3), PlotTweak.getYPosition(current_axes, -0.25),
                                  "Frequency", size=8, rotation=90)

    def figure_rwp_last_hour(self):
        name = "figure_rwp_last_hour"
        self.figures[name] = Figure(self.figureFolder,name, figsize = [self.figureWidth, 4],
                                    ncols=2, nrows=2,
                                    bottom=0.11, top=0.87,
                                    hspace=0.08, wspace=0.24,
                                    )
        fig = self.figures[name]
        blue_color = Colorful.getDistinctColorList("blue")
        red_color = Colorful.getDistinctColorList("red")
        for ind, trainingSet in enumerate(self.trainingSetList):
            current_axes = fig.getAxes(ind)
            dataframe = self.completeDataFrame[trainingSet]
            dataframe_filtered = self.completeDataFrameFiltered[trainingSet]

            filt_rwp = dataframe_filtered["rwp_last_hour"]*1e3
            orig_rwp = dataframe["rwp_last_hour"]*1e3
            print("rwp", trainingSet, filt_rwp.max(), orig_rwp.max())
            orig_rwp.plot.hist(ax=current_axes, color = red_color )
            filt_rwp.plot.hist(ax=current_axes, color = blue_color)

        # tweaks
        start = 0.0
        end = 150
        interval = 25
        ticks = numpy.arange(0, end + interval, interval)
        tickLabels = [f"{t:d}" for t in ticks]

        showList = Data.cycleBoolean(len(ticks))

        showList[0] = True
        showList[-1] = False
        for ind, trainingSet in enumerate(self.trainingSetList):
            current_axes = fig.getAxes(ind)

            if "4" in trainingSet:
                yticks = numpy.arange(0, 151, 25)
                ytickLabels = [f"{t:d}" for t in yticks]

            else:
                yticks = numpy.arange(0, 501, 100)
                ytickLabels = [f"{t:d}" for t in yticks]


            yshowList = Data.cycleBoolean(len(yticks))
            yshowList[-1] =False




            current_axes.set_xlim([start, end])
            PlotTweak.setYaxisLabel(current_axes,"")




            current_axes.set_xticks(ticks)
            current_axes.set_xticklabels(tickLabels)
            PlotTweak.hideLabels(current_axes.xaxis, showList)

            current_axes.set_yticks(yticks)
            current_axes.set_yticklabels(ytickLabels)
            PlotTweak.hideLabels(current_axes.yaxis, yshowList)

            if ind in [0,1]:
                PlotTweak.hideXTickLabels(current_axes)

            PlotTweak.setAnnotation(current_axes,
                                    self.annotationCollection[trainingSet],
                                    xPosition=PlotTweak.getXPosition(current_axes, 0.05),
                                    yPosition = PlotTweak.getYPosition(current_axes, 0.9))

            if ind == 0:
                collectionOfLabelsColors = {"RWP last hour average (LES filtered data)" : blue_color,
                                            "RPW last hour average (all LES data)" : red_color,}
                legendLabelColors = PlotTweak.getPatches(collectionOfLabelsColors)

                artist = current_axes.legend(handles=legendLabelColors,
                                             loc=(0.0, 1.03),
                                             frameon=True,
                                             framealpha=1.0,
                                             ncol=1)

                current_axes.add_artist(artist)

            if ind == 3:
                current_axes.text(PlotTweak.getXPosition(current_axes, -0.3), PlotTweak.getYPosition(current_axes, -0.25),
                                  "(" + PlotTweak.getLatexLabel("g\ m^{-2}") + ")", size=8)
            if ind == 0:
                current_axes.text(PlotTweak.getXPosition(current_axes, -0.3), PlotTweak.getYPosition(current_axes, -0.25),
                                  "Frequency", size=8, rotation=90)

def main():
    try:
        locationsFile = sys.argv[1]
    except IndexError:
        locationsFile = "/home/aholaj/mounttauskansiot/puhtiwork/EmulatorManuscriptData_revision_v0.1/locationsMounted.yaml"

    figObject = FiguresRevision(locationsFile)

    figObject.figure_relative_relative_change_lwp()
    figObject.figure_surface_precipitation_accumulated()
    figObject.figure_rwp_last_hour()
    figObject.figure_lwp_scatter_plot()

    figObject.finalise()


if __name__ == "__main__":
    start = time.time()
    now = datetime.now().strftime('%d.%m.%Y %H.%M')
    print(f"Script started {now}.")
    main()
    end = time.time()
    now = datetime.now().strftime('%d.%m.%Y %H.%M')
    print(f"Script completed {now} in {Data.timeDuration(end - start)}")