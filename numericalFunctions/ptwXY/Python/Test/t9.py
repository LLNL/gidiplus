#! /usr/bin/env python

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

"""
This stuff should be repeatable.
"""

import random

from numericalFunctions import pointwiseXY_C

class t( pointwiseXY_C.pointwiseXY_C ) :

    def __init__( self, n, overflowSize = 10, xMin = -1, xMax = 1, yMin = 0, yMax = 100 ) :

        pointwiseXY_C.pointwiseXY_C.__init__( self, initialSize = n, overflowSize = 10 )
        for i in range( n ) :
            r = random.random( )
            x = xMin * r + xMax * ( 1. - r )
            r = random.random( )
            y = yMin * r + yMax * ( 1. - r )
            self.setValue( x, y )

random.seed( 314159 )
a = t( 5 )
b = t( 6 )

print( a )
print( b )
if( a[0][0] < b[0][0] ) :
    b.setValue( a[0][0], 0 )
else :
    a.setValue( b[0][0], 0 )

if( a[-1][0] > b[-1][0] ) :
    b.setValue( a[-1][0], 0 )
else :
    a.setValue( b[-1][0], 0 )

print( a )
print( b )
print( a + b )
