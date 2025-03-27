# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

from xData import enums as enumsModule
from xData import constant as constantModule
from xData import XYs1d as XYs1dModule
from xData import regions as regionsModule
from xData import series1d as series1dModule

epsilon = 1e-6

dataset1 = [1 - 2 * epsilon, 1,
            1 -     epsilon, 1 + 3 * epsilon, 
            1              , 1 + 8 * epsilon]

dataset2 = [1,                 1,
            1 + 1.5 * epsilon, 1 - 2 * epsilon, 
            1 +   4 * epsilon, 1 - 5 * epsilon]

regions1d = regionsModule.Regions1d()
for data in [dataset1, dataset2]:
    regions1d.append(XYs1dModule.XYs1d(data=data, dataForm='list'))

regions1d.saveToFile('Data/regions1d.xml')

dataset1 = [1.0e-4, 1,  1, 20]
dataset2 = [1, 22,  1e3, 40]
for otherInterpolation in [enumsModule.Interpolation.linlin, enumsModule.Interpolation.flat, enumsModule.Interpolation.loglog]:
    regions1d = regionsModule.Regions1d()
    interpolation = enumsModule.Interpolation.linlin
    for index, data in enumerate([dataset1, dataset2]):
        if index != 0:
            interpolation = otherInterpolation
        regions1d.append(XYs1dModule.XYs1d(data=data, dataForm='list', interpolation=interpolation))
    regions1d.saveToFile('Data/regions1d.2.%s.xml' % interpolation)

constantModule.Constant1d(41, 1e3, 1e4).saveToFile('Data/constand1d.xml')

Legendre = series1dModule.LegendreSeries([1, .3, .01])
Legendre.saveToFile('Data/Legendre1d.xml')

polymonial = series1dModule.Polynomial1d([1, .3, .01, .2], 1, 10)
polymonial.saveToFile('Data/polynomial1d.xml')
