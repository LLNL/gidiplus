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

os.system( 'cd %s; make -s clean; ./groupTestAll1 -v' % CPATH )

def getIntegerValue( name, ls ) :

    s = "# %s = " % name
    n = len( s )
    if( ls[0][:n] != s ) : raise Exception( '%s: %s does not contain %s info: "%s"' % ( __file__, name, ls[0][:-1] ) )
    value = int( ls[0].split( '=' )[1] )
    return( ls[1:], value )

def compareValues( label, i, v1, v2 ) :

    sv1, sv2 = '%.12e' % v1, '%.12e' % v2
    sv1, sv2 = '%.7e' % float( sv1 ), '%.7e' % float( sv2 )
    if( sv1 != sv2 ) : print( '<%s> <%s>' % ( sv1, sv2 ) )
    if( sv1 != sv2 ) : raise Exception( '%s: values %e and %e diff by %e at %d for label = %s' % ( __file__, v1, v2, v2 - v1, i, label ) )

def compareGroups( fileName, norm, g1 ) :

    label = fileName + '_' + norm
    g2 = getXData( label )
    if( len( g1 ) != len( g2 ) ) : raise Exception( '%s: for %s len( g1 ) = %d != len( g2 ) = %d' %( __file__, label, len( g1 ), len( g2 ) ) )
    for i , g1X in enumerate( g1 ) : compareValues( label, i, g1X, g2[i] )

def getXData( fileName ) :

    fileName_ = os.path.join( CPATH, fileName + '.dat' )
    f = open( fileName_ )
    ls = f.readlines( )
    f.close( )
    ls, length = getIntegerValue( 'length', ls )
    if( len( ls ) != length ) : raise Exception( '%s: len( ls ) = %s != length = %d for file %s' % ( len( ls ), length, fileName ) )
    data = [ float( l ) for l in ls ]
    return( data )
    
def getXYData( fileName ) :

    fileName_ = os.path.join( CPATH, fileName )
    f = open( fileName_ )
    ls = f.readlines( )
    f.close( )
    ls, length = getIntegerValue( 'length', ls )
    data = [ list( map( float, l.split( ) ) ) for l in ls ]
    data = pointwiseXY_C.pointwiseXY_C( data, initialSize = len( data ), overflowSize = 10 )
    return( data )

def checkOneFunctionGrouping( fileName, groupBoundaries ) :

    flux = getXYData( fileName + '.dat' )
    flux_None = flux.groupOneFunction( groupBoundaries )
    compareGroups( fileName, 'None', flux_None )
    flux_dx = flux.groupOneFunction( groupBoundaries, norm = 'dx' )
    compareGroups( fileName, 'dx', flux_dx )
    flux_norm = flux.groupOneFunction( groupBoundaries, norm = flux_None )
    compareGroups( fileName, 'norm', flux_norm )
    return( flux, flux_None )

def checkTwoFunctionGrouping( fileName, groupBoundaries, flux, flux_None ) :

    crossSection = getXYData( fileName + '.dat' )
    crossSection_None = crossSection.groupTwoFunctions( groupBoundaries, flux )
    compareGroups( fileName, 'None', crossSection_None )
    crossSection_dx = crossSection.groupTwoFunctions( groupBoundaries, flux, norm = 'dx' )
    compareGroups( fileName, 'dx', crossSection_dx )
    crossSection_norm = crossSection.groupTwoFunctions( groupBoundaries, flux, norm = flux_None )
    compareGroups( fileName, 'norm', crossSection_norm )
    return( crossSection )

def checkThreeFunctionGrouping( fileName, groupBoundaries, flux, crossSection, flux_None ) :

    multiplicity = getXYData( fileName + '.dat' )
    multiplicity_None = multiplicity.groupThreeFunctions( groupBoundaries, flux, crossSection )
    compareGroups( fileName, 'None', multiplicity_None )
    multiplicity_dx = multiplicity.groupThreeFunctions( groupBoundaries, flux, crossSection, norm = 'dx' )
    compareGroups( fileName, 'dx', multiplicity_dx )
    multiplicity_norm = multiplicity.groupThreeFunctions( groupBoundaries, flux, crossSection, norm = flux_None )
    compareGroups( fileName, 'norm', multiplicity_norm )

groupBoundaries = getXData( 'groupBoundaries' )
flux, flux_None = checkOneFunctionGrouping( 'flux', groupBoundaries )
crossSection = checkTwoFunctionGrouping( 'crossSection', groupBoundaries, flux, flux_None )
checkThreeFunctionGrouping( 'multiplicity', groupBoundaries, flux, crossSection, flux_None )
