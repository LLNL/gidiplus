# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

import os
import argparse

description = """
Compares each point in file1 to file2. If a y-value in file1 is not the same as the y-value in file2, the y-value in file2 is 
compared to *offset + slope * y1* where y1 is the y-value in file1. If these y-values differ, information about the differences
are printed.
"""

parser = argparse.ArgumentParser(description = description)

parser.add_argument('file1',                                               help = 'The base file.')
parser.add_argument('file2',                                               help = 'The modified file.')
parser.add_argument('--offset', type = float, default = 0.0,               help = 'The value of the offset.')
parser.add_argument('--slope',  type = float, default = 1.0,               help = 'The value of the slope.')

args = parser.parse_args()

def getData(lines) :

    reactionDatas = ''.join(lines).split('# Reaction: ')[1:]

    reactions = []
    for reactionData in reactionDatas :
        sectionDatas = reactionData.split('#')
        
        label = sectionDatas.pop(0).split(os.sep)[0].strip()

        sections = []
        for sectionData in sectionDatas:
            style, data = sectionData.split('::')
            style = style.strip( )

            points = []
            lines = data.split('\n')
            for line in lines:
                point = line.strip()
                if len(point) == 0: continue
                points.append( list(map(float,point.split())))

            sections.append([ style, points ])

        reactions.append([ label, sections ])

    return reactions

offset = args.offset
slope  = args.slope

with open(args.file1) as file: lines1 = file.readlines()
with open(args.file2) as file: lines2 = file.readlines()

if len(lines1) != len(lines2): raise Exception('Input files do not have the same length.')

reactions1 = getData(lines1)
reactions2 = getData(lines2)

if len(reactions1) != len(reactions2): raise Exception('Data in the input files do not start at the same point: %s vs. %s.' % ( start1, start2 ))

for reactionIndex, reaction1 in enumerate(reactions1):
    label1, sections1 = reaction1
    label2, sections2 = reactions2[reactionIndex]

    printLabel = True

    if label1 != label2: raise Exception('label1 = "%s" != label2 = "%s.' % ( label1, label2 ))
    if len(sections1) != len(sections2): raise Exception('len(sections1) %s != len(sections2) = %s.' % ( len(sections1) != len(sections2) ))

    for sectionIndex, section1 in enumerate(sections1):
        style1, points1 = section1
        style2, points2 = sections2[sectionIndex]

        printStyle = True

        if style1 != style2: raise Exception('style1 = "%s" != style2 = "%s.' % ( style1, style2 ))
        if len(sections1) != len(sections2): raise Exception('len(sections1) %s != len(sections2) = %s.' % ( len(sections1) != len(sections2) ))

        diffCounter = 0
        for pointIndex, point1 in enumerate(points1):
            x1, y1 = point1
            x2, y2 = points2[pointIndex]
            if x1 != x2: raise Exception('x1 = %s != x2 = %s.' % ( x1, x2))
            if y1 != y2:
                y1p = offset + slope * y1
                if abs(y1p - y2) > 1e-6 * ( y1p + y2 ):
                    if printLabel: print(label1)
                    printLabel = False
                    if printStyle: print('    %s' % style1)
                    printStyle = False
                    diffCounter += 1
                    if diffCounter < 5: print( '        Difference at x = %s, %.16s vs %.16s.' % ( x1, y1p, y2 ))
        if diffCounter > 0: print('        *** %s differences found.' % diffCounter)
