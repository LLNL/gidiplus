# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

from __future__ import print_function

import os

from numericalFunctions import pointwiseXY_C

accuracy = 1e-2
biSectionMax = 0.

if( 'CHECKOPTIONS' in os.environ ) :
    options = os.environ['CHECKOPTIONS'].split( )
    if( '-e' in options ) : print( __file__ )

CPATH = '../../../../Test/Flat/toLinear'

os.system( 'cd %s; make -s clean; ./flatToLinear -v > v' % CPATH )

def skipBlankLines( ls ) :

    i = 0
    for i, l in enumerate( ls ) :
        if( l.strip( ) != '' ) : break
    ls = ls[i:]
    if( ( len( ls ) == 1 ) and ( ls[0].strip( ) == '' ) ) : ls = []
    return( ls )

def getIntegerValue( name, ls ) :

    s = "# %s = " % name
    n = len( s )
    if( ls[0][:n] != s ) : raise Exception( '%s: missing %s info: "%s"' % ( __file__, name, ls[0][:-1] ) )
    value = int( ls[0].split( '=' )[1] )
    return( ls[1:], value )

def getDoubleValue( name, ls ) :

    s = "# %s = " % name
    n = len( s )
    if( ls[0][:n] != s ) : raise Exception( '%s: missing %s info: "%s"' % ( __file__, name, ls[0][:-1] ) )
    value = float( ls[0].split( '=' )[1] )
    return( ls[1:], value )

def compareValues( label, i, v1, v2 ) :

    sv1, sv2 = '%.12e' % v1, '%.12e' % v2
    sv1, sv2 = '%.7e' % float( sv1 ), '%.7e' % float( sv2 )
    if( sv1 != sv2 ) : print( '<%s> <%s>' % ( sv1, sv2 ) )
    if( sv1 != sv2 ) : raise Exception( '%s: values %e and %e diff by %e at %d for label = %s' % ( __file__, v1, v2, v2 - v1, i, label ) )

def getXYData( ls, biSectionMax, accuracy ) :

    ls, length = getIntegerValue( 'length', ls )
    data = [ list( map( float, ls[i].split( ) ) ) for i in range( length ) ]
    data = pointwiseXY_C.pointwiseXY_C( data, initialSize = len( data ), overflowSize = 10, biSectionMax = biSectionMax, accuracy = accuracy, safeDivide = True, interpolation = "flat" )
    ls = ls[length:]
    ls = skipBlankLines( ls )
    return( ls, data )

def compareXYs( XYs1, XYs2, label ) :

    if( len( XYs1 ) != len( XYs2 ) ) : raise Exception( 'for %s: len( XYs1 ) = %s != len( XYs2 ) = %s' % ( label, len( XYs1 ), len( XYs2 ) ) )
    for i, xy in enumerate( XYs1 ) :
        compareValues( "x division " + label, count, xy[0], XYs2[i][0] )
        compareValues( "y division " + label, count, xy[1], XYs2[i][1] )

def toLinearCheck( count, ls ) :

    ls, epsm = getDoubleValue( 'epsm', ls )
    ls, epsp = getDoubleValue( 'epsp', ls )
    ls, XYs = getXYData( ls, biSectionMax, accuracy )
    ls, resultsC = getXYData( ls, biSectionMax, accuracy )
    results = XYs.changeInterpolation( 'lin-lin', lowerEps = epsm, upperEps = epsp )
    compareXYs( resultsC, results, "toLinearCheck" )
    return( ls )

f = open( os.path.join( CPATH, 'v' ) )
ls = f.readlines( )
f.close( )

ls, accuracy = getDoubleValue( 'accuracy', ls )

count = 0
while( len( ls ) ) :
    count += 1
    ls = toLinearCheck( count, ls )
