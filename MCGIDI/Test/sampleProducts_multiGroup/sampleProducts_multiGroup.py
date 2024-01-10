# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

import sys
import os

if( not( os.path.exists( 'Outputs' ) ) ) : os.mkdir( 'Outputs' )

output = ''
cmd = ''
sep = './'
for arg in sys.argv[1:] :
    cmd += sep + arg
    sep = ' '
    if( os.sep in arg ) : continue
    output += '.' + arg
output = output[1:]

benchmark = os.path.join( 'Benchmarks', output + '.out' ).replace( '..', '.' )
output = os.path.join( 'Outputs', output + '.out' ).replace( '..', '.' )

if( os.system( cmd + ' > ' + output ) != 0 ) : raise Exception( 'FAILURE: %s' % cmd )
cmd = '../Utilities/diff.com sampleProducts_multiGroup/%s %s %s' % ( sys.argv[1], benchmark, output )
os.system( cmd )
