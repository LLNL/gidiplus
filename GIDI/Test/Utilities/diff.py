# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

from __future__ import print_function
import sys

relativeMax = 0.0

def readFile( fileName ) :

    fIn = open( fileName )
    lines = fIn.readlines( )
    fIn.close( )

    lines[:-1]
    return( lines )

def error( ) :

    print( "FAILURE: %s: difference in output" % sys.argv[1] )
    sys.exit( 0 )

def errorFloats( float1, float2, i1, i2 ) :

    global relativeMax

    diff = float1 - float2
    biggest = max( abs( float1 ), abs( float2 ) )
    relative = diff / biggest
    if( len( sys.argv ) > 4 ) : print( 'float1 = %.14e  float2 = %.14e  diff = % 10.2e  rel = % 10.2e: %6s %6s' % ( float1, float2, diff, relative, i1, i2 ) )
    if( len( sys.argv ) < 6 ) : error( )
    if( abs( relative ) > abs( relativeMax ) ) : relativeMax = relative

lines1 = readFile( sys.argv[2] )[1:]
lines2 = readFile( sys.argv[3] )[1:]

for i1, line1 in enumerate( lines1 ) :
    line2 = lines2[i1]
    if( line1 != line2 ) :
        if( ( ":::" in line1 ) and ( ":::" in line1 ) ) : continue
        if( ( "/.../" in line1 ) and ( "/.../" in line1 ) ) : continue
        split1 = line1.split( "::" )
        if( len( split1 ) != 2 ) : error( )

        split2 = line2.split( "::" )
        if( len( split1 ) != len( split2 ) ) : error( )

        if( split1[0] != split2[0]  ) : error( )

        floats1 = list( map( float, split1[1].split( ) ) )
        floats2 = list( map( float, split2[1].split( ) ) )

        if( len( floats1 ) != len( floats2 ) ) : error( )

        for i2, float1 in enumerate( floats1 ) :
            float2 = floats2[i2]
            if( float1 != float2 ) :
                if( abs( float1 - float2 ) > 1e-6 * ( abs( float1 ) + abs( float2 ) ) ) : errorFloats( float1, float2, i1 + 2, i2 )

if( relativeMax != 0.0 ) : print( 'relativeMax = ', relativeMax )
