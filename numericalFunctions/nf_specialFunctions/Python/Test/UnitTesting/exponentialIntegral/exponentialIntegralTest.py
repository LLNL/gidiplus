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

f = open( '../../../../Test/UnitTesting/exponentialIntegral/exponentialIntegralTest.dat' )
ls = f.readlines( )
f.close( )

errors = 0
for l in ls :
    if( ';' in l ) : break
    if( ',' not in l ) : continue
    n, x, f = l.split( '{' )[1].split( '}' )[0].split( ',' )
    n = int( n )
    x = float( x )
    f = float( f )
    En = specialFunctions.exponentialIntegral( n, x )
    r = 1
    if( f != 0 ) : r = En / f - 1
    if( abs( r ) > 3e-14 ) :
        print( n, x, f, En, r )
        errors += 1

if( errors > 0 ) : raise Exception( "FAILED: %s" % __file__ )
