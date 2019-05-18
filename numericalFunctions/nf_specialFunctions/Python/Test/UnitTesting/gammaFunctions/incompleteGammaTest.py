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

f = open( '../../../../Test/UnitTesting/gammaFunctions/incompleteGammaTest.dat' )
ls = f.readlines( )
f.close( )

errors = 0
for l in ls :
    if( ';' in l ) : break
    if( ',' not in l ) : continue
    s, x, f = l.split( '{' )[1].split( '}' )[0].split( ',' )
    s = float( s )
    x = float( x )
    f = float( f )
    gamma = specialFunctions.incompleteGamma( s, x )
    r = 1
    if( f != 0 ) :
        r = gamma / f - 1
    else :
        if( gamma == 0 ) : r = 0
    if( abs( r ) > 1e-13 ) :
        print( x, f, gamma, r )
        errors += 1

if( errors > 0 ) : raise Exception( "FAILED: %s" % __file__ )
