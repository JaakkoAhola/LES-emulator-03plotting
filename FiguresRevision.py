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
        ticks = numpy.arange(start, end + .01, 0.5)
        tickLabels = [f"{t:.1f}" for t in ticks]

        showList = Data.cycleBoolean(len(ticks))

        showList[0] = True
        showList[-1] = False
        for ind, trainingSet in enumerate(self.trainingSetList):
            current_axes = fig.getAxes(ind)

            if "4" in trainingSet:
                yticks = numpy.arange(start, 151, 25)
                ytickLabels = [f"{t:.0f}" for t in yticks]

            else:
                yticks = numpy.arange(start, 501, 100)
                ytickLabels = [f"{t:.0f}" for t in yticks]

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

            dataframe_filtered = self.completeDataFrameFiltered[trainingSet]

            dataframe_filtered.plot.scatter(x="lwp",
                                            y="total_water_path_last_hour_mean",
                                            ax=current_axes,
                                            color = blue_color,
                                            alpha=0.3)

        # tweaks
        start = 0.0
        end = 800
        interval = 100
        ticks = numpy.arange(start, end + .01, interval)
        tickLabels = [f"{t:.0f}" for t in ticks]

        showList = Data.cycleBoolean(len(ticks))

        showList[0] = True
        showList[-1] = False
        for ind, trainingSet in enumerate(self.trainingSetList):
            current_axes = fig.getAxes(ind)

            current_axes.set_xlim([start, end])
            current_axes.axline([0, 0], [1, 1], color="k")
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
                ticks = numpy.arange(start, end + interval, interval)
                tickLabels = [f"{t:.0e}" for t in ticks]
                tickLabels[0] = "0"

                yticks = numpy.arange(start, 151, 25)
                ytickLabels = [f"{t:.0f}" for t in yticks]

            else:
                start = 0.0
                end = 4e-3
                interval = 1e-3
                ticks = numpy.arange(start, end + interval, interval)
                tickLabels = [f"{t:.0e}" for t in ticks]
                tickLabels[0] = "0"

                yticks = numpy.arange(start, 501, 100)
                ytickLabels = [f"{t:.0f}" for t in yticks]


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
        ticks = numpy.arange(start, end + interval, interval)
        tickLabels = [f"{t:.0f}" for t in ticks]

        showList = Data.cycleBoolean(len(ticks))

        showList[0] = True
        showList[-1] = False
        for ind, trainingSet in enumerate(self.trainingSetList):
            current_axes = fig.getAxes(ind)

            if "4" in trainingSet:
                yticks = numpy.arange(start, 151, 25)
                ytickLabels = [f"{t:.0f}" for t in yticks]

            else:
                yticks = numpy.arange(start, 501, 100)
                ytickLabels = [f"{t:.0f}" for t in yticks]


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

    def figure_cloud_top_scatter_plot(self):
        name = "figure_cloud_top_scatter_plot"
        self.figures[name] = Figure(self.figureFolder,name, figsize = [self.figureWidth, 4],
                                    ncols=2, nrows=2,
                                    hspace=0.08, wspace=0.12,
                                    bottom=0.11, top=0.87,
                                    left=0.15
                                    )

        fig = self.figures[name]
        blue_color = Colorful.getDistinctColorList("blue")
        red_color = Colorful.getDistinctColorList("red")
        for ind, trainingSet in enumerate(self.trainingSetList):
            current_axes = fig.getAxes(ind)

            dataframe = self.completeDataFrame[trainingSet]
            mask = ~dataframe[self.filterIndex]

            filtered_out = deepcopy(dataframe[mask])

            dataframe_filtered = self.completeDataFrameFiltered[trainingSet]

            disapeared_cloud_index = filtered_out[filtered_out["zc_end_value"]<0].index

            filtered_out.loc[disapeared_cloud_index, "zc_end_value"] = 0

            filtered_out.plot.scatter(x="zc_first",
                                      y="zc_end_value",
                                      ax=current_axes,
                                      color=red_color,
                                      alpha=0.3)

            dataframe_filtered.plot.scatter(x="zc_first",
                                            y="zc_end_value",
                                            ax=current_axes,
                                            color = blue_color,
                                            alpha=0.3)

            current_axes.axline([0, 0], [1, 1], color="k")

        # tweaks
        start = 0.0
        end = 3000
        interval = 500
        ticks = numpy.arange(start, end + .01, interval)
        tickLabels = [f"{t:.0f}" for t in ticks]

        showList = Data.cycleBoolean(len(ticks))

        showList[0] = True
        showList[-1] = False
        for ind, trainingSet in enumerate(self.trainingSetList):
            current_axes = fig.getAxes(ind)

            current_axes.set_xlim([start, end])
            current_axes.set_ylim([start, end])
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
                collectionOfLabelsColors = {"Cloud top change" : blue_color,
                                            "Cloud top change (filtered out)" : red_color,
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
                                   PlotTweak.getUnitLabel("Cloud\ top\ height\ in\ the\ beginning", "m"), size=8)
            if ind == 0:
                current_axes.text(PlotTweak.getXPosition(current_axes, -0.33), PlotTweak.getYPosition(current_axes, -0.8),
                                  PlotTweak.getUnitLabel("Cloud\ top\ height\ in\ the\ end", "m"), size=8, rotation=90)


    def figure_cloud_base_scatter_plot(self):
        name = "figure_cloud_base_scatter_plot"
        self.figures[name] = Figure(self.figureFolder,name, figsize = [self.figureWidth, 4],
                                    ncols=2, nrows=2,
                                    hspace=0.08, wspace=0.12,
                                    bottom=0.11, top=0.87,
                                    left=0.15
                                    )

        fig = self.figures[name]
        blue_color = Colorful.getDistinctColorList("blue")
        red_color = Colorful.getDistinctColorList("red")
        for ind, trainingSet in enumerate(self.trainingSetList):
            current_axes = fig.getAxes(ind)

            dataframe = self.completeDataFrame[trainingSet]
            mask = ~dataframe[self.filterIndex]

            filtered_out = deepcopy(dataframe[mask])

            dataframe_filtered = self.completeDataFrameFiltered[trainingSet]

            disapeared_cloud_index = filtered_out[filtered_out["zb_end_value"]<0].index

            filtered_out.loc[disapeared_cloud_index, "zb_end_value"] = 0

            filtered_out.plot.scatter(x="zb_first",
                                      y="zb_end_value",
                                      ax=current_axes,
                                      color=red_color,
                                      alpha=0.3)

            dataframe_filtered.plot.scatter(x="zb_first",
                                            y="zb_end_value",
                                            ax=current_axes,
                                            color = blue_color,
                                            alpha=0.3)

            current_axes.axline([0, 0], [1, 1], color="k")

        # tweaks
        start = 0.0
        end = 3000
        interval = 500
        ticks = numpy.arange(start, end + .01, interval)
        tickLabels = [f"{t:.0f}" for t in ticks]

        showList = Data.cycleBoolean(len(ticks))

        showList[0] = True
        showList[-1] = False
        for ind, trainingSet in enumerate(self.trainingSetList):
            current_axes = fig.getAxes(ind)

            current_axes.set_xlim([start, end])
            current_axes.set_ylim([start, end])
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
                collectionOfLabelsColors = {"Cloud base change" : blue_color,
                                            "Cloud base change (filtered out)" : red_color,
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
                                   PlotTweak.getUnitLabel("Cloud\ base\ height\ in\ the\ beginning", "m"), size=8)
            if ind == 0:
                current_axes.text(PlotTweak.getXPosition(current_axes, -0.33), PlotTweak.getYPosition(current_axes, -0.8),
                                  PlotTweak.getUnitLabel("Cloud\ base\ height\ in\ the\ end", "m"), size=8, rotation=90)


    def figure_mixed_layer_cloud_thickness_vs_cloud_top_height_scatter_plot(self):
        name = "figure_mixed_layer_cloud_thickness_vs_cloud_top_height_scatter_plot"
        self.figures[name] = Figure(self.figureFolder,name, figsize = [self.figureWidth, 4],
                                    ncols=2, nrows=2,
                                    hspace=0.08, wspace=0.12,
                                    bottom=0.11, top=0.87,
                                    left=0.15
                                    )

        fig = self.figures[name]
        blue_color = Colorful.getDistinctColorList("blue")
        red_color = Colorful.getDistinctColorList("red")
        for ind, trainingSet in enumerate(self.trainingSetList):
            current_axes = fig.getAxes(ind)

            dataframe = self.completeDataFrame[trainingSet]
            mask = ~dataframe[self.filterIndex]

            filtered_out = deepcopy(dataframe[mask])

            dataframe_filtered = self.completeDataFrameFiltered[trainingSet]

            disapeared_cloud_index = filtered_out[filtered_out["zc_end_value"]<0].index

            filtered_out.loc[disapeared_cloud_index, "zc_last_hour"] = 0
            filtered_out.loc[disapeared_cloud_index, "delta_zm"] = 0

            filtered_out.plot.scatter(x="zc_last_hour",
                                      y="delta_zm",
                                      ax=current_axes,
                                      color=red_color,
                                      alpha=0.3)

            dataframe_filtered.plot.scatter(x="zc_last_hour",
                                            y="delta_zm",
                                            ax=current_axes,
                                            color = blue_color,
                                            alpha=0.3)


        # tweaks
        start = 0.0
        end = 3000
        interval = 500
        ticks = numpy.arange(start, end + .01, interval)
        tickLabels = [f"{t:.0f}" for t in ticks]

        showList = Data.cycleBoolean(len(ticks))

        showList[0] = True
        showList[-1] = False


        ystart = 0.0
        yend = 1250
        yinterval = 250
        yticks = numpy.arange(ystart, yend + .01, yinterval)
        ytickLabels = [f"{t:.0f}" for t in yticks]

        yshowList = Data.cycleBoolean(len(ticks))

        yshowList[0] = True
        yshowList[-1] = False

        for ind, trainingSet in enumerate(self.trainingSetList):
            current_axes = fig.getAxes(ind)

            current_axes.set_xlim([start, end])
            current_axes.set_ylim([ystart, yend])
            PlotTweak.setYaxisLabel(current_axes,"")
            PlotTweak.setXaxisLabel(current_axes,"")



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
                collectionOfLabelsColors = {"Cloud base change" : blue_color,
                                            "Cloud base change (filtered out)" : red_color,
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
                                   PlotTweak.getUnitLabel("Cloud\ top\ height\ average\ last\ hour", "m"), size=8)
            if ind == 0:
                current_axes.text(PlotTweak.getXPosition(current_axes, -0.33), PlotTweak.getYPosition(current_axes, -0.8),
                                  PlotTweak.getUnitLabel("Mixed\ Layer\ cloud\ thickness\ last\ hour", "m"), size=8, rotation=90)

    def figure_mixed_layer_cloud_thickness_histogram(self):
        name = "figure_mixed_layer_cloud_thickness_histogram"
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
            dataframe_filtered["delta_zm"].plot.hist(ax=current_axes, color = blue_color)


        # tweaks
        start = 0.0
        end = 1250
        interval = 250
        ticks = numpy.arange(start, end + interval, interval)
        tickLabels = [f"{t:.0f}" for t in ticks]

        showList = Data.cycleBoolean(len(ticks))

        showList[0] = True
        showList[-1] = False
        for ind, trainingSet in enumerate(self.trainingSetList):
            current_axes = fig.getAxes(ind)

            if "4" in trainingSet:
                yticks = numpy.arange(0, 40, 10)
                ytickLabels = [f"{t:d}" for t in yticks]

            else:
                yticks = numpy.arange(0, 126, 25)
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
            if ind == 3:
                current_axes.text(PlotTweak.getXPosition(current_axes, -0.7), PlotTweak.getYPosition(current_axes, -0.25),
                                   PlotTweak.getUnitLabel("Mixed\ Layer\ cloud\ thickness\ last\ hour", "m"), size=8)

    def figure_wpos_tendency(self):
        name = "figure_wpos_tendency"
        self.figures[name] = Figure(self.figureFolder,name, figsize = [self.figureWidth, 4],
                                    ncols=2, nrows=2,
                                    hspace=0.08, wspace=0.12,
                                    bottom=0.13, top=0.90,
                                    )

        fig = self.figures[name]
        blue_color = Colorful.getDistinctColorList("blue")
        for ind, trainingSet in enumerate(self.trainingSetList):
            current_axes = fig.getAxes(ind)
            dataframe_filtered = self.completeDataFrameFiltered[trainingSet]

            dataframe_filtered["WposLastHourTendency"].plot.hist(ax=current_axes, color = blue_color)

        start = -0.2
        end = -start
        interval = 0.05
        ticks = numpy.arange(start, end + interval, interval)
        tickLabels = [f"{t:.2f}" for t in ticks]
        tickLabels[tickLabels.index("-0.00")] = "0"

        showList = Data.cycleBoolean(len(ticks))

        showList[0] = False
        showList[-1] = False
        for ind, trainingSet in enumerate(self.trainingSetList):
            current_axes = fig.getAxes(ind)

            if "4" in trainingSet:
                yticks = numpy.arange(0, 55, 10)
                ytickLabels = [f"{t:d}" for t in yticks]

            else:
                yticks = numpy.arange(0, 301, 25)
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

            PlotTweak.setXTickSizes(current_axes, Data.cycleBoolean(len(ticks)))
            PlotTweak.setYTickSizes(current_axes, Data.cycleBoolean(len(yticks)))


            PlotTweak.setAnnotation(current_axes,
                                    self.annotationCollection[trainingSet],
                                    xPosition=PlotTweak.getXPosition(current_axes, 0.05),
                                    yPosition = PlotTweak.getYPosition(current_axes, 0.9))

            if ind == 0:
                collectionOfLabelsColors = {"""Domain mean updraft velocity at cloud base
change during last hour""" : blue_color,
                                            #"LWP relative change (all LES data)" : red_color,
                                            }
                legendLabelColors = PlotTweak.getPatches(collectionOfLabelsColors)

                artist = current_axes.legend(handles=legendLabelColors,
                                             loc=(0.25, 1.03),
                                             frameon=True,
                                             framealpha=1.0,
                                             ncol=1)

                current_axes.add_artist(artist)

            if ind == 0:
                current_axes.text(PlotTweak.getXPosition(current_axes, -0.25), PlotTweak.getYPosition(current_axes, -0.25),
                                  "Frequency", size=8, rotation=90)
            if ind == 3:
                current_axes.text(PlotTweak.getXPosition(current_axes, -0.35), PlotTweak.getYPosition(current_axes, -0.25),
                                   PlotTweak.getUnitLabel(r"\frac{d(w_{pos.dom.})}{dt}", "m s^{-1} h^{-1}"), size=8)


    def figure_temperature_decoupled(self):
        name = "figure_temperature_decoupled"
        self.figures[name] = Figure(self.figureFolder,name, figsize = [self.figureWidth, 4],
                                    ncols=2, nrows=2,
                                    hspace=0.08, wspace=0.12,
                                    bottom=0.11, top=0.93,
                                    )

        fig = self.figures[name]
        blue_color = Colorful.getDistinctColorList("blue")
        for ind, trainingSet in enumerate(self.trainingSetList):
            current_axes = fig.getAxes(ind)
            dataframe_filtered = self.completeDataFrameFiltered[trainingSet]
            temperature = dataframe_filtered["temperature_decoupled"]
            print(f"{trainingSet} temperature decoupled, min: {temperature.min():.5f}, max: {temperature.max():.5f}", )
            temperature.plot.hist(ax=current_axes,
                                  color=blue_color)

        start = 0.0
        end = 1.5
        interval = 0.25
        ticks = numpy.arange(start, end + interval, interval)
        tickLabels = [f"{t:.2f}" for t in ticks]
        tickLabels[0] = "0"

        showList = Data.cycleBoolean(len(ticks))
        # showList[0] = False
        # showList[-1] = False
        for ind, trainingSet in enumerate(self.trainingSetList):
            current_axes = fig.getAxes(ind)

            if "4" in trainingSet:
                yend=60
                yticks = numpy.arange(0, yend+1, 10)
                ytickLabels = [f"{t:d}" for t in yticks]

            else:
                yend=200
                yticks = numpy.arange(0, yend+1, 25)
                ytickLabels = [f"{t:d}" for t in yticks]

            yshowList = Data.cycleBoolean(len(yticks))


            current_axes.set_xlim([start, end])
            current_axes.set_ylim([0, yend])
            PlotTweak.setYaxisLabel(current_axes,"")

            current_axes.set_xticks(ticks)
            current_axes.set_xticklabels(tickLabels)


            if ind in [0,1]:
                PlotTweak.hideXTickLabels(current_axes)
            if ind in [1,3]:
                PlotTweak.hideYTickLabels(current_axes)


            PlotTweak.hideLabels(current_axes.xaxis, showList)

            current_axes.set_yticks(yticks)
            current_axes.set_yticklabels(ytickLabels)
            PlotTweak.hideLabels(current_axes.yaxis, yshowList)

            PlotTweak.setXTickSizes(current_axes, Data.cycleBoolean(len(ticks)))
            PlotTweak.setYTickSizes(current_axes, Data.cycleBoolean(len(yticks)))


            PlotTweak.setAnnotation(current_axes,
                                    self.annotationCollection[trainingSet],
                                    xPosition=PlotTweak.getXPosition(current_axes, 0.05),
                                    yPosition = PlotTweak.getYPosition(current_axes, 0.9))

            if ind == 0:
                collectionOfLabelsColors = {"Temperature difference" : blue_color,

                                            }
                legendLabelColors = PlotTweak.getPatches(collectionOfLabelsColors)

                artist = current_axes.legend(handles=legendLabelColors,
                                             loc=(0.65, 1.03),
                                             frameon=True,
                                             framealpha=1.0,
                                             ncol=1)

                current_axes.add_artist(artist)

            if ind == 0:
                current_axes.text(PlotTweak.getXPosition(current_axes, -0.25),
                                  PlotTweak.getYPosition(current_axes, -0.25),
                                  "Frequency",
                                  size=8,
                                  rotation=90)
            if ind == 3:
                current_axes.text(PlotTweak.getXPosition(current_axes, -0.35),
                                  PlotTweak.getYPosition(current_axes, -0.25),
                                  PlotTweak.getUnitLabel(r"\theta_{l,top}-\theta_{l,bot}", "K"),
                                  size=8)


    def figure_water_decoupled(self):
        name = "figure_water_decoupled"
        self.figures[name] = Figure(self.figureFolder,name, figsize = [self.figureWidth, 4],
                                    ncols=2, nrows=2,
                                    hspace=0.08, wspace=0.12,
                                    bottom=0.11, top=0.93,
                                    )

        fig = self.figures[name]
        blue_color = Colorful.getDistinctColorList("blue")
        for ind, trainingSet in enumerate(self.trainingSetList):
            current_axes = fig.getAxes(ind)
            dataframe_filtered = self.completeDataFrameFiltered[trainingSet]
            water = dataframe_filtered["water_decoupled"]
            water.plot.hist(ax=current_axes, color = blue_color)

        start = -0.5
        end = 0.1
        interval = .1
        ticks = numpy.arange(start, end + interval, interval)
        tickLabels = [f"{t:.1f}" for t in ticks]


        showList = Data.cycleBoolean(len(ticks), startBoolean = False)

        # showList[0] = False
        # showList[-1] = False
        return
        for ind, trainingSet in enumerate(self.trainingSetList):
            current_axes = fig.getAxes(ind)

            if "4" in trainingSet:
                yticks = numpy.arange(0, 71, 10)
                ytickLabels = [f"{t:d}" for t in yticks]

            else:
                yticks = numpy.arange(0, 201, 25)
                ytickLabels = [f"{t:d}" for t in yticks]

            yshowList = Data.cycleBoolean(len(yticks))
            yshowList[-1] =True

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

            PlotTweak.setXTickSizes(current_axes, Data.cycleBoolean(len(ticks)))
            PlotTweak.setYTickSizes(current_axes, Data.cycleBoolean(len(yticks)))


            PlotTweak.setAnnotation(current_axes,
                                    self.annotationCollection[trainingSet],
                                    xPosition=PlotTweak.getXPosition(current_axes, 0.05),
                                    yPosition = PlotTweak.getYPosition(current_axes, 0.9))

            if ind == 0:
                collectionOfLabelsColors = {"Temperature difference" : blue_color,

                                            }
                legendLabelColors = PlotTweak.getPatches(collectionOfLabelsColors)

                artist = current_axes.legend(handles=legendLabelColors,
                                             loc=(0.65, 1.03),
                                             frameon=True,
                                             framealpha=1.0,
                                             ncol=1)

                current_axes.add_artist(artist)

            if ind == 0:
                current_axes.text(PlotTweak.getXPosition(current_axes, -0.25), PlotTweak.getYPosition(current_axes, -0.25),
                                  "Frequency", size=8, rotation=90)
            if ind == 3:
                current_axes.text(PlotTweak.getXPosition(current_axes, -0.35), PlotTweak.getYPosition(current_axes, -0.25),
                                   PlotTweak.getUnitLabel(r"q_{bot}-q_{top}", "g\ kg^{-1}"), size=8)
    def figure_decoupled_scatter(self):
        name = "figure_decoupled_scatter"
        self.figures[name] = Figure(self.figureFolder,name, figsize = [self.figureWidth, 4],
                                    ncols=2, nrows=2,
                                    hspace=0.08, wspace=0.12,
                                    bottom=0.15, top=0.93,
                                    left = 0.15, right=0.97,
                                    )

        fig = self.figures[name]
        green_color = Colorful.getDistinctColorList("green")
        for ind, trainingSet in enumerate(self.trainingSetList):
            current_axes = fig.getAxes(ind)
            dataframe_filtered = self.completeDataFrameFiltered[trainingSet]

            dataframe_filtered.plot.scatter(x="temperature_decoupled",
                                            y="water_decoupled",
                                            ax=current_axes,
                                            c="prcp",
                                            colormap='viridis',
                                            alpha=0.3)

            current_axes.axhline(y=0.5, c="b")
            current_axes.axvline(x=0.5, c="r")

        start = 0.
        end = 1.5
        interval = 0.25
        ticks = numpy.arange(start, end + interval, interval)
        tickLabels = [f"{t:.1f}" for t in ticks]
        tickLabels[0] = "0"

        showList = Data.cycleBoolean(len(ticks))

        # showList[0] = False
        # showList[-1] = False
        ystart = 0
        yend = 1.5
        yinterval = 0.25
        yticks = numpy.arange(ystart, yend + yinterval, yinterval)
        ytickLabels = [f"{t:.1f}" for t in yticks]
        ytickLabels[0] = "0"

        for ind, trainingSet in enumerate(self.trainingSetList):
            current_axes = fig.getAxes(ind)
            PlotTweak.setYaxisLabel(current_axes,"")
            PlotTweak.setXaxisLabel(current_axes,"")

            yshowList = Data.cycleBoolean(len(yticks))
            yshowList[-1] =True

            current_axes.set_xlim([start, end])
            current_axes.set_ylim([ystart, yend])
            PlotTweak.setYaxisLabel(current_axes,"")

            current_axes.set_xticks(ticks)
            current_axes.set_xticklabels(tickLabels)
            PlotTweak.hideLabels(current_axes.xaxis, showList)

            current_axes.set_yticks(yticks)
            current_axes.set_yticklabels(ytickLabels)
            PlotTweak.hideLabels(current_axes.yaxis, yshowList)

            if ind in [0,1]:
                PlotTweak.hideXTickLabels(current_axes)
            if ind in [1,3]:
                PlotTweak.hideYTickLabels(current_axes)

            PlotTweak.setXTickSizes(current_axes, Data.cycleBoolean(len(ticks)))
            PlotTweak.setYTickSizes(current_axes, Data.cycleBoolean(len(yticks)))


            PlotTweak.setAnnotation(current_axes,
                                    self.annotationCollection[trainingSet],
                                    xPosition=PlotTweak.getXPosition(current_axes, 0.05),
                                    yPosition = PlotTweak.getYPosition(current_axes, 0.9))

            if ind == 0:
                collectionOfLabelsColors = {"Water & temperature difference" : green_color,

                                            }
                legendLabelColors = PlotTweak.getPatches(collectionOfLabelsColors)

                artist = current_axes.legend(handles=legendLabelColors,
                                             loc=(0.45, 1.03),
                                             frameon=True,
                                             framealpha=1.0,
                                             ncol=1)

                current_axes.add_artist(artist)

            if ind == 0:
                current_axes.text(PlotTweak.getXPosition(current_axes, -0.35),
                                  PlotTweak.getYPosition(current_axes, -0.25),
                                  PlotTweak.getUnitLabel(r"q_{bot}-q_{top}", "g\ kg^{-1}"),
                                  size=8,
                                  rotation=90)
            if ind == 3:
                current_axes.text(PlotTweak.getXPosition(current_axes, -0.35),
                                  PlotTweak.getYPosition(current_axes, -0.35),
                                  PlotTweak.getUnitLabel(r"\theta_{l,top}-\theta_{l,bot}", "K"),
                                  size=8)

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
    figObject.figure_cloud_top_scatter_plot()
    figObject.figure_cloud_base_scatter_plot()
    figObject.figure_mixed_layer_cloud_thickness_vs_cloud_top_height_scatter_plot()
    figObject.figure_mixed_layer_cloud_thickness_histogram()
    figObject.figure_wpos_tendency()
    figObject.figure_temperature_decoupled()
    figObject.figure_water_decoupled()
    figObject.figure_decoupled_scatter()


    figObject.finalise()


if __name__ == "__main__":
    start = time.time()
    now = datetime.now().strftime('%d.%m.%Y %H.%M')
    print(f"Script started {now}.")
    main()
    end = time.time()
    now = datetime.now().strftime('%d.%m.%Y %H.%M')
    print(f"Script completed {now} in {Data.timeDuration(end - start)}")