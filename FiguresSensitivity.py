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


class FiguresSensitivity(ManuscriptFigures):

    def __init__(self, dict_of_locationsFiles):

        self.dict_of_locationsFiles = dict_of_locationsFiles

        super().__init__(dict_of_locationsFiles["softLimit"])

        self.sensitivity_sets = {}

        for key in dict_of_locationsFiles:
            self.sensitivity_sets[key] = ManuscriptFigures(dict_of_locationsFiles[key])

        self.filter_names = list(self.dict_of_locationsFiles)
        self.rmse = {}

        for ind, trainingSet in enumerate(self.trainingSetList):
            self.rmse[trainingSet] = {}
            for predictor in self.predictorShortNames:
                self.rmse[trainingSet][self.predictorClearNames[predictor]] = numpy.zeros(len(self.filter_names))


    def collect_sensitivity_stats(self):
        for ind, trainingSet in enumerate(self.trainingSetList):
            for predictor in self.predictorShortNames:
                self.rmse[trainingSet][self.predictorClearNames[predictor]] = [ self.sensitivity_sets[key].statsCollection[trainingSet].loc[predictor]["rmse"] for key in self.dict_of_locationsFiles]


    def figure_sensitivity(self):
        name = "figure_sensitivity"
        self.figures[name] = Figure(self.figureFolder,name, figsize = [self.figureWidth, 4],
                                    ncols=2, nrows=2,
                                    hspace=0.08, wspace=0.12,
                                    bottom=0.11, top=0.93,
                                    )

        fig = self.figures[name]


        for ind, trainingSet in enumerate(self.trainingSetList):
            current_axes = fig.getAxes(ind)
            data = self.rmse[trainingSet]
            index = ["NA", "Soft", "Hard"]
            df = pandas.DataFrame(data, index=index)
            df.plot.bar(rot=0, ax=current_axes, color = self.predictorColors.values())





def main():
    try:
        dict_of_locationsFiles = sys.argv[1]
    except IndexError:
        dict_of_locationsFiles = {"noLimit": "/home/aholaj/mounttauskansiot/puhtiwork/EmulatorManuscriptData_revision_v0.1_noFilter/locationsMounted.yaml",
                                  "softLimit": "/home/aholaj/mounttauskansiot/puhtiwork/EmulatorManuscriptData_revision_v0.1/locationsMounted.yaml",
                                  "hardLimit": "/home/aholaj/mounttauskansiot/puhtiwork/EmulatorManuscriptData_revision_v0.1_hardLimit/locationsMounted.yaml",
                                  }

    fs = FiguresSensitivity(dict_of_locationsFiles)
    fs.collect_sensitivity_stats()
    print(fs.rmse)

    fs.figure_sensitivity()

    fs.finalise()

if __name__ == "__main__":
    start = time.time()
    now = datetime.now().strftime('%d.%m.%Y %H.%M')
    print(f"Script started {now}.")
    main()
    end = time.time()
    now = datetime.now().strftime('%d.%m.%Y %H.%M')
    print(f"Script completed {now} in {Data.timeDuration(end - start)}")