# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

from __future__ import print_function

import os
import sys

from numericalFunctions import pointwiseXY_C

def getOptions( fileName = None ) :

    options = []
    if( 'CHECKOPTIONS' in os.environ ) : options = os.environ['CHECKOPTIONS'].split( )
    for argv in sys.argv[1:] :
        if( argv[0] == '-' ) : options.append( argv )
    if( ( fileName is not None ) and ( '-e' in options ) ) : print( fileName )
    return( options )

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
    if( ls[0][:n] != s ) : raise Exception( 'missing %s info: "%s"' % ( name, ls[0][:-1] ) )
    value = int( ls[0].split( '=' )[1] )
    return( ls[1:], value )

def getDoubleValue( name, ls ) :

    s = "# %s = " % name
    n = len( s )
    if( ls[0][:n] != s ) : raise Exception( 'missing %s info: "%s"' % ( name, ls[0][:-1] ) )
    value = float( ls[0].split( '=' )[1] )
    return( ls[1:], value )

def getStringValue( name, ls ) :

    s = "# %s = " % name
    n = len( s )
    if( ls[0][:n] != s ) : raise Exception( 'missing %s info: "%s"' % ( name, ls[0][:-1] ) )
    value = ls[0].split( '=' )[1].strip( )
    return( ls[1:], value )

def getXData( ls, sep = None ) :

    ls, length = getIntegerValue( 'length', ls )
    Xs = []
    for i in range( length ) :
        l = ls[i]
        if( l[0] == '#' ) : l = l[1:]
        if( sep is not None ) : l = l.split( sep )[1]
        Xs.append( float( l ) )
    return( ls[length:], Xs )

def getXYData( ls ) :

    ls = skipBlankLines( ls )
    ls, length = getIntegerValue( 'length', ls )
    data = [ list( map( float, ls[i].split( ) ) ) for i in range( length ) ]
    data = pointwiseXY_C.pointwiseXY_C( data, initialSize = len( data ), overflowSize = 10 )
    ls = ls[length:]
    ls = skipBlankLines( ls )
    return( ls, data )

def compareValues( label, count, v1, v2 ) :
    
    sv1, sv2 = '%.12e' % v1, '%.12e' % v2
    sv1, sv2 = '%.7e' % float( sv1 ), '%.7e' % float( sv2 )
    if( sv1 != sv2 ) : print( '<%s> <%s>' % ( sv1, sv2 ) )
    if( sv1 != sv2 ) : raise Exception( 'values %e and %e diff by %e at count = %d for label = %s' % ( v1, v2, v2 - v1, count, label ) )

def compareXYs( count, label, xys1, xys2 ) :

    for i, xy in enumerate( xys1 ) :
        compareValues( label + ' x-values', count, xy[0], xys2[i][0] )
        compareValues( label + ' y-values', count, xy[1], xys2[i][1] )
