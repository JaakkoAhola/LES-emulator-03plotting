#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 17.1.2022

@author: Jaakko Ahola, Finnish Meteorological Institute
@licence: MIT licence Copyright
"""
import numpy
import os
import sys
import time
from datetime import datetime
from copy import deepcopy
import pandas

sys.path.append(os.environ["LESMAINSCRIPTS"])
from Colorful import Colorful
from Data import Data
from Figure import Figure
from PlotTweak import PlotTweak


sys.path.append("../LES-emulator-02postpros")
from plotEMUL import ManuscriptFigures
from itertools import repeat

class FiguresSensitivity(ManuscriptFigures):

    def __init__(self, dict_of_locationsFiles):

        self.dict_of_locationsFiles = dict_of_locationsFiles

        super().__init__(dict_of_locationsFiles["softLimit"])

        self.sensitivity_sets = {}

        for key in dict_of_locationsFiles:
            self.sensitivity_sets[key] = ManuscriptFigures(dict_of_locationsFiles[key])

        self.filter_names = list(self.dict_of_locationsFiles)

        self.stats_names = ["rmse", "rSquared"]
        self.stats_proper_names = dict(zip(self.stats_names, ["RMSE", PlotTweak.getLatexLabel('R^2')]))
        self.sens_stats = dict(zip(self.stats_names, repeat(None)))


        for stat_name in self.stats_names:
            self.sens_stats[stat_name] = {}
            for ind, trainingSet in enumerate(self.trainingSetList):
                self.sens_stats[stat_name][trainingSet] = {}
                for predictor in self.predictorShortNames:
                    self.sens_stats[stat_name][trainingSet][self.predictorClearNames[predictor]] = numpy.zeros(len(self.filter_names))

        self._collect_stats()

    def _collect_stats(self):
        for stat_name in self.stats_names:
            for ind, trainingSet in enumerate(self.trainingSetList):
                for predictor in self.predictorShortNames:
                    self.sens_stats[stat_name][trainingSet][self.predictorClearNames[predictor]] = [ self.sensitivity_sets[key].statsCollection[trainingSet].loc[predictor][stat_name] for key in self.dict_of_locationsFiles]


    def figure_sensitivity_rmse(self):
        self._figure_sensitivity("rmse",
                                        [0, 0.12, 0.02],
                                        2)



    def figure_sensitivity_rSquared(self):
        self._figure_sensitivity("rSquared",
                                        [0, 1, 0.2],
                                        1)


    def _figure_sensitivity(self, stat_name, ytick_specs, float_decimals):
        name = "figure_sensitivity_" + stat_name
        self.figures[name] = Figure(self.figureFolder,name, figsize = [self.figureWidth, 4],
                                    ncols=2, nrows=2,
                                    hspace=0.08, wspace=0.12,
                                    bottom=0.11, top=0.93,
                                    left = 0.15, right=0.97
                                    )

        fig = self.figures[name]

        ystart = ytick_specs[0]
        yend = ytick_specs[1]
        yinterval = ytick_specs[2]

        yticks = numpy.arange(ystart, yend + yinterval, yinterval)
        ytickLabels = [f"{t:.{float_decimals}f}" for t in yticks]
        yshowList = Data.cycleBoolean(len(yticks))
        yshowList[-1]=False

        for ind, trainingSet in enumerate(self.trainingSetList):

            current_axes = fig.getAxes(ind)
            data = self.sens_stats[stat_name][trainingSet]
            index = ["NA", "Soft", "Hard"]
            df = pandas.DataFrame(data, index=index)
            df.plot.bar(rot=0,
                        ax=current_axes,
                        color=self.predictorColors.values(),
                        legend=False)

            current_axes.set_yticks(yticks)
            current_axes.set_yticklabels(ytickLabels)
            PlotTweak.hideLabels(current_axes.yaxis, yshowList)

            PlotTweak.setAnnotation(current_axes, self.annotationCollection[trainingSet],
                                    xPosition=PlotTweak.getXPosition(current_axes, 0.20),
                                    yPosition=PlotTweak.getYPosition(current_axes, 0.90))

            if ind in [0,1]:
                PlotTweak.hideXTickLabels(current_axes)
            if ind in [1,3]:
                PlotTweak.hideYTickLabels(current_axes)

            if ind == 1:
                collectionOfLabelsColors = dict(zip(list(self.predictorClearNames.values()), list(self.predictorColors.values())))
                legendLabelColors = PlotTweak.getPatches(collectionOfLabelsColors)

                artist = current_axes.legend(handles=legendLabelColors,
                                              loc=(-0.6, 1.05),
                                              frameon=True,
                                              framealpha=1.0,
                                              ncol=3)

                current_axes.add_artist(artist)

            if ind == 0:
                current_axes.text(PlotTweak.getXPosition(current_axes, -0.30),
                                  PlotTweak.getYPosition(current_axes, -0.05),
                                  self.stats_proper_names[stat_name],
                                  size=8,
                                  rotation=90)

def main():
    try:
        dict_of_locationsFiles = sys.argv[1]
    except IndexError:
        dict_of_locationsFiles = {"noLimit": "/home/aholaj/mounttauskansiot/puhtiwork/EmulatorManuscriptData_revision_v0.1_noFilter/locationsMounted.yaml",
                                  "softLimit": "/home/aholaj/mounttauskansiot/puhtiwork/EmulatorManuscriptData_revision_v0.1/locationsMounted.yaml",
                                  "hardLimit": "/home/aholaj/mounttauskansiot/puhtiwork/EmulatorManuscriptData_revision_v0.1_hardLimit/locationsMounted.yaml",
                                  }

    fs = FiguresSensitivity(dict_of_locationsFiles)

    fs.figure_sensitivity_rmse()
    fs.figure_sensitivity_rSquared()

    fs.finalise()

if __name__ == "__main__":
    start = time.time()
    now = datetime.now().strftime('%d.%m.%Y %H.%M')
    print(f"Script started {now}.")
    main()
    end = time.time()
    now = datetime.now().strftime('%d.%m.%Y %H.%M')
    print(f"Script completed {now} in {Data.timeDuration(end - start)}")
