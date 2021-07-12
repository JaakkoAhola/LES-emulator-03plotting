#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 17:23:47 2020

@author: Jaakko Ahola, Finnish Meteorological Institute
@licence: MIT licence Copyright
"""

import matplotlib
import numpy
import os
import pandas
import pathlib
import seaborn
import scipy
import sys
import time

sys.path.append(os.environ["LESMAINSCRIPTS"])
from Colorful import Colorful
from Data import Data
from InputSimulation import InputSimulation
from Figure import Figure
from Plot import Plot
from PlotTweak import PlotTweak

class MethodFigures:

    def __init__(self, folder, figurefolder):
            self.folder = pathlib.Path(folder)
            self.figurefolder = figurefolder

            self.soundIN = pandas.read_csv(self.folder / "sound_in", delim_whitespace=True, header=None, names = ["height", "temperature", "water", "u-wind", "v-wind"])

            self.design = pandas.read_csv(self.folder.parent / "design.csv")

            self.designInd = int(self.folder.name[-3:])-1



    def figureProfileExample(self):



        fig = Figure(self.figurefolder,"figureProfileExample", figsize=[8.3/2.54, 2.5], ncols = 2, nrows = 1, wspace =0.15, bottom = 0.18, left=0.20)
        print(fig.getFigSize())
        colorList = [Colorful.getDistinctColorList("red"), Colorful.getDistinctColorList("blue")]
        variables = ["temperature", "water"]

        tempRef = None
        for ind,temp in enumerate(self.soundIN["temperature"].values):
            if tempRef is None:
                tempRef = temp
            else:
                if numpy.abs( tempRef - temp) > numpy.finfo(float).eps*10:
                    pblhIND = ind-1
                    break
        pblh = self.soundIN["height"].iloc[pblhIND]

        tpot_inv = self.design["tpot_inv"].iloc[self.designInd]
        tpot_pbl = self.design["tpot_pbl"].iloc[self.designInd]

        r_t = self.soundIN["water"].max()
        q_inv = self.design["q_inv"].iloc[self.designInd]

        t_grad = 0.3
        invThi = tpot_inv / t_grad

        print("pblh", pblh, self.soundIN["height"].max())

        maxheight = Data.roundUpToScale(self.soundIN["height"].max(), 100)
        print("pblh", pblh, self.soundIN["height"].max(), maxheight)

        minTemp = Data.roundDown(self.soundIN["temperature"].min(), 1)
        maxTemp = Data.roundUp(self.soundIN["temperature"].max(), 1)


        minWater = 0
        maxWaterAxes = Data.roundUp(self.soundIN["water"].max(), 1)

        print("maxWater", self.soundIN["water"].max())
        lineWidthProfile = 1.4
        lineWidthDash = 1.4
        lineWidthArrow  = 1.4
        annotationYCoord = 0.87
        for ind in range(2):
            ax = fig.getAxes(ind)

            self.soundIN.plot(ax=ax,x=variables[ind], y="height", color = colorList[ind], legend = False, linewidth = lineWidthProfile)

            ax.set_ylim([0, maxheight])

            Plot.getHorizontalLine(ax, pblh)
            if ind == 0:
                PlotTweak.setXaxisLabel(ax,"\\theta_{L}", "K")
                PlotTweak.setYaxisLabel(ax,"Altitude", "m")



                xticks = PlotTweak.setXticks(ax, start = minTemp, end = maxTemp, interval = 1, integer = True)

                xShownLabelsBoolean = PlotTweak.setXLabels(ax, xticks, start = minTemp, end = maxTemp, interval = 5, integer = True)

                PlotTweak.setXTickSizes(ax, xShownLabelsBoolean)
                ax.set_xlim([minTemp, maxTemp])

                xmin = ax.get_xlim()[0]
                xmax = ax.get_xlim()[1]

                ymin = ax.get_ylim()[0]
                ymax = ax.get_ylim()[1]

                k_rate = (ymax-ymin)/(xmax-xmin)

                xPoint = (xmax-xmin)*0.2+xmin
                yPoint = k_rate*(xPoint-tpot_pbl)

                PlotTweak.setAnnotation(ax, "(a) Temperature\nprofile", xPosition =0.03, yPosition =annotationYCoord, xycoords="axes fraction")

                ax.axvline(tpot_pbl + tpot_inv , color = "k" , linestyle = "--" , ymax = (pblh + invThi)/ymax, linewidth=lineWidthDash) 

                # ax.arrow(xPoint, yPoint, tpot_pbl-xPoint, 0-yPoint,
                #           facecolor='black', shape = "full", linewidth = param, head_length = param*1.5*3., head_width = 3.*param, overhang = 0.1, head_starts_at_zero = True, length_includes_head = True)
                ax.annotate(PlotTweak.getMathLabel("tpot_pbl"),
                            xy=(tpot_pbl, 0),
                            xytext = (xPoint, yPoint),
                            arrowprops=dict(facecolor='black', arrowstyle = "->", linewidth = lineWidthArrow),
                            horizontalalignment='left',
                            verticalalignment='bottom'
                            )
                #ax.text((xPoint-tpot_pbl)*0.6 + tpot_pbl, (yPoint- ymin)*0.3 + ymin,  PlotTweak.getMathLabel("tpot_pbl"))
                arrowParam = 0.25
                pblhArrowX = (xmax - (tpot_pbl + tpot_inv))*arrowParam + (tpot_pbl + tpot_inv)
                pblhArrowXText = (xmax - (tpot_pbl + tpot_inv))*arrowParam*1.5 + (tpot_pbl + tpot_inv)
                ax.annotate("",
                            xy=( pblhArrowX, pblh),
                            xytext=(pblhArrowX, 0),
                            arrowprops = dict(arrowstyle = "<->",facecolor='black', linewidth = lineWidthArrow))
                ax.text(pblhArrowXText, pblh*0.43,  PlotTweak.getMathLabel("pblh"), rotation =90)

                ax.text(tpot_pbl+tpot_inv*0.40, pblh*0.41,  PlotTweak.getMathLabel("tpot_inv"))
                ax.annotate("",
                            xy=(tpot_pbl, pblh*0.5),
                            xytext=(tpot_pbl + tpot_inv, pblh*0.5),
                            arrowprops = dict(arrowstyle = "<->",facecolor='black', linewidth = lineWidthArrow))

            else:


                PlotTweak.setXaxisLabel(ax,"r_t", "g\ kg^{-1}")
                PlotTweak.hideYTickLabels(ax)

                xticks = numpy.arange(minWater, maxWaterAxes + 0.001, 1)
                xlabels = [f"{t:.1f}" for t in xticks]

                xShowList = Data.getMaskedList(xticks, xticks[0::4])

                ax.set_xticks(xticks)
                ax.set_xticklabels(xlabels)
                PlotTweak.setXTickSizes(ax, xShowList)
                PlotTweak.hideLabels(ax.xaxis, xShowList)
                ax.set_xlim([minWater, maxWaterAxes])

                xmin = ax.get_xlim()[0]
                xmax = ax.get_xlim()[1]

                ymin = ax.get_ylim()[0]
                ymax = ax.get_ylim()[1]

                PlotTweak.setAnnotation(ax, "(b) Humidity\nprofile", xPosition = 0.20, yPosition = annotationYCoord, xycoords="axes fraction" )

                ax.axvline(r_t - q_inv , color = "k" , linestyle = "--" , ymax = (pblh + invThi)/ymax)

                k_rate = (ymax-ymin)/(xmin-xmax)

                xPoint = self.soundIN["water"].max()*.8
                yPoint = k_rate*(xPoint-self.soundIN["water"].max())

                ax.annotate(PlotTweak.getLatexLabel("r_t"),
                            xy=(self.soundIN["water"].max()*0.99, 0),
                            xytext = (xPoint, yPoint),
                            arrowprops=dict(facecolor='black', arrowstyle = "->", linewidth = lineWidthArrow),
                            horizontalalignment='left',
                            verticalalignment='bottom'
                            )
                ax.annotate("",
                            xy=(r_t-q_inv, pblh*0.5),
                            xytext=(r_t, pblh*0.5),
                            arrowprops = dict(arrowstyle = "<->",facecolor='black', linewidth = lineWidthArrow))
                ax.text(r_t-q_inv*0.60, pblh*0.41,  PlotTweak.getMathLabel("q_inv"))

            yticks = PlotTweak.setYticks(ax, start= 0, end = maxheight, interval = 100, integer=True)

            yShownLabelsBoolean = PlotTweak.setYLabels(ax, yticks, start= 0, end = maxheight, interval = 500)

            PlotTweak.setYTickSizes(ax, yShownLabelsBoolean)

        fig.save()

if __name__ == "__main__":
    start = time.time()

    figObject = MethodFigures( "/home/aholaj/mounttauskansiot/puhtiwork/eclair_training_simulations/case_emulator_DESIGN_v3.2_LES_ECLAIR_branch_ECLAIRv2.0.cray.fast_LVL4_night/emul040",
                                  "/home/aholaj/mounttauskansiot/puhtiwork/EmulatorManuscriptData_3/figures")

    figObject.figureProfileExample()
    end = time.time()
    print("Script completed in " + str(round((end - start),0)) + " seconds")
