# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

import os

from numericalFunctions import pointwiseXY_C

if( 'CHECKOPTIONS' in os.environ ) :
    options = os.environ['CHECKOPTIONS'].split( )
    if( '-e' in options ) : print( __file__ )

CPATH = '../../../../Test/UnitTesting/integrate'

os.system( 'cd %s; make -s clean; ./integrationWithWeight_x -v > v' % CPATH )

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

def getXYData( ls ) :

    ls, length = getIntegerValue( 'length', ls )
    ls, interpolation = getIntegerValue( 'interpolation', ls )
    interpolationStr = [ 'lin-lin', 'log-lin', 'lin-log', 'log-log', 'flat' ][interpolation]
    data = [ list( map( float, ls[i].split( ) ) ) for i in range( length ) ]
    data = pointwiseXY_C.pointwiseXY_C( data, initialSize = len( data ), overflowSize = 10, interpolation = interpolationStr )
    ls = ls[length:]
    ls = skipBlankLines( ls )
    return( ls, data )

def checkIntegration( count, xMin, xMax, data, sum ) :

    V = data.integrateWithWeight_x( xMin, xMax )
    compareValues( 'sum', count, V, sum )
    invV = data.integrateWithWeight_x( xMax, xMin )
    if( V != -invV ) : raise Exception( '%s: at %d V = %g != -invV = %g' % ( __file__, count, V, invV ) )

f = open( os.path.join( CPATH, 'v' ) )
ls = f.readlines( )
f.close( )

count = 0
while( len( ls ) ) :
    ls = skipBlankLines( ls )
    if( len( ls ) ) :
        if( '# groupTwoFunctions testing' in ls[0] ) : break
    count += 1
    ls, xMin = getDoubleValue( 'xMin', ls )
    ls, xMax = getDoubleValue( 'xMax', ls )
    ls, data = getXYData( ls )
    ls = skipBlankLines( ls )
    l, ls = ls[0], ls[1:]
    sum = float( l.split( 'sum = ' )[1].split( )[0] )
    checkIntegration( count, xMin, xMax, data, sum )
