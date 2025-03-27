# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

import sys

with open( sys.argv[2] ) as file : lines1 = file.readlines( )
with open( sys.argv[3] ) as file : lines2 = file.readlines( )

errs = [ 0.0, 0.0, 1e-6, 0.0, 1e-8, 0.0, 0.0 ]

def test( ) :

    for index1, line1 in enumerate( lines1 ) :
        line2 = lines2[index1]
        if( line1 != line2 ) :
            if( 'energy =' in line1 ) :
                if( 'energy =' in line2 ) :
                    doubles1 = map( float, line1.split( '=' )[-1].split( ) )
                    doubles2 = map( float, line2.split( '=' )[-1].split( ) )
                    for index2, (double1, double2) in enumerate( zip(doubles1, doubles2) ) :
                        if( double1 != double2 ) :
                            err = abs( double1 - double2 ) / max( abs( double1 ), abs( double2 ) )
                            if( err > errs[index2] ) : return( 1 )
                else :
                    return( 1 )
            else :
                return( 1 )

    return( 0 )

if( test( ) != 0 ) : print( 'FAILURE: %s: difference in output' % sys.argv[1] )
