# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

import sys

def readFile( fileName ) :

    fIn = open( fileName )
    lines = fIn.readlines( )
    fIn.close( )

    return( lines )

lines1 = readFile( sys.argv[2] )
lines2 = readFile( sys.argv[3] )

def printAndExit( ) :
    print( "FAILURE: %s: difference in output" % sys.argv[1] )
    sys.exit( 0 )

if( len( lines1 ) != len( lines2 ) ) : printAndExit( )

for i1, line1 in enumerate( lines1 ) :
    line2 = lines2[i1]

    if( line1 != line2 ) :
        if( ( 'KE = ' in line1 ) and ( 'KE = ' in line2 ) ) :
            value1 = float( line1.split( 'KE = ' )[1] )
            value2 = float( line2.split( 'KE = ' )[1] )
            if( abs( value1 - value2 ) < 1e-5 * ( abs( value1 ) + abs( value2 ) ) ) : continue
            printAndExit( )
