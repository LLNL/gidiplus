# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

from __future__ import print_function
import sys
import os

from argparse import ArgumentParser

code = 'deposition_multiGroup'

description = """Runs %s on a list of targets for defined options.""" % code

parser = ArgumentParser( description = description )
parser.add_argument( '-v', '--verbose', action = 'count', default = 0,      help = 'Enable verbose output.' )
parser.add_argument( 'targets', action = 'append',                          help = 'List of target IDs to check.' )

args = parser.parse_args( )

def checkOptions( target, options ) :

    if( args.verbose > 0 ) : print()
    output = '%s.%s' % ( code, target )
    for option in options : output += '.%s' % option
    output = os.path.join( 'Outputs', output + '.out' )
    output = output.replace( '..', '.' )
    cmd = './%s %s --tid %s > o' % ( code, ' '.join( options ), target )
    if( args.verbose > 0 ) : print( cmd )
    os.system( cmd )
    cmd = '../Utilities/diff.com %s/%s %s o' % ( code, code, output )
    if( args.verbose > 0 ) : print( cmd )
    os.system( cmd )

def checkTarget( target ) :

    for options in [ [ '' ], [ '-p' ] ] : checkOptions( target, options )

for target in args.targets : checkTarget( target )
