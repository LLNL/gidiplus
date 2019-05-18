#
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
#

from __future__ import print_function

import os
import sys

from numericalFunctions import specialFunctions

options = []
if( 'CHECKOPTIONS' in os.environ ) : options = os.environ['CHECKOPTIONS'].split( )
for argv in sys.argv[1:] : options += [ argv ]

if( '-e' in options ) : print( __file__ )

x = 1e-4
while( x < 5 ) :
    erf  = specialFunctions.erf( x )
    erfc = specialFunctions.erf( x, True )
    print( x, erf, erfc, erf+ erfc )
    x *= 1.2
