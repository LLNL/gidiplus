#! /usr/bin/env python3

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

from __future__ import print_function

from argparse import ArgumentParser

description = """Compares three output files from activeReactions. The first must be the output from running activeReactions with no -r options and no --invert option.
    The second and third must be run with the same -r options but one must also have the --invert option and the other must not."""

parser = ArgumentParser( description = description )

parser.add_argument( 'label',                   help = 'A string to print out if an error is found that indicates the test code.' )
parser.add_argument( 'file1', type = open,      help = 'An output file from "activeReactions" with no "-r" option and no "--invert" option.' )
parser.add_argument( 'file2', type = open,      help = 'An output file from "activeReactions" some "-r" options.' )
parser.add_argument( 'file3', type = open,      help = 'An output file from "activeReactions" the same "-r" options the the prior argument but not the same "--invert" option.' )

args = parser.parse_args( )

def getData( file ) :

    lines = file.readlines( )
    file.close( )

    keys = []
    data = {}
    for lineNumber, line in enumerate( lines ) :
        if( '::' in line ) :
            key, values = line.split( "::" )
            key = key.strip( )
            if( key == "" ) : key = priorKey
            if( key not in keys ) : keys.append( key )
            if( key not in data ) : data[key] = []
            data[key].append( list( map( float, values.split( ) ) ) )
        else :
            priorKey = line.strip( )

    return( keys, data )

keys1, file1 = getData( args.file1 )
dummy, file2 = getData( args.file2 )
dummy, file3 = getData( args.file3 )

for key in keys1 :
    data1 = file1[key]
    if( key not in file2 ) :
        data2 = len( data1 ) * [ [] ]
    else :
        data2 = file2[key]

    if( key not in file3 ) :
        data3 = len( data1 ) * [ [] ]
    else :
        data3 = file3[key]

    for i1, values1 in enumerate( data1 ) :
        values2 = data2[i1]
        values3 = data3[i1]

        if( len( values2 ) == 0 ) : values2 = len( values1 ) * [ 0.0 ]
        if( len( values3 ) == 0 ) : values3 = len( values1 ) * [ 0.0 ]

        for i2, value in enumerate( values1 ) :
            sum_2_3 = values2[i2] + values3[i2]
            diff = value - sum_2_3
            ratio = diff
            if( value != 0.0 ) : ratio /= value
            if( abs( ratio ) > 1e-8 ) :
                print( '    Bad sum for data "%s" at index %s: %18.9e %18.9e %18.9e %10.2e %9.1e' % ( key, i2, value, values2[i2], values3[i2], diff, ratio ) )
