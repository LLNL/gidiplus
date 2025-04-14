# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

import sys

from xData import XYs1d as XYs1dModule
from xData import regions as regionsModule

with open(sys.argv[1]) as fIn:
    lines = fIn.readlines()

curves = []
counter = 0
mode = 0
for line in lines:
    if mode == 0:
        if '# regions = ' in line:
            path = line.split('=')[1].strip()
            regions1d = regionsModule.Regions1d.readXML_file(path)
            regions1d.plotLabel = 'regions1d'
            curves.append(regions1d)
            mode = 1
    else:
        if counter == 0:
            if '# length = ' in line:
                counter = int(line.split(';')[0].split('=')[1].strip())
                epsilons = list(map(float, (line.split(';')[1].split())))
                data = []
        else:
            data.append(list(map(float, line.split())))
            counter -= 1
            if counter == 0:
                curve = XYs1dModule.XYs1d(data=data)
                curve.plotLabel = "%s" % epsilons
                curves.append(curve)

curves[0].multiPlot(curves)
