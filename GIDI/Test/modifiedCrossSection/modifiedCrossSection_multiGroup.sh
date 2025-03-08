#! /usr/bin/env bash

file=$1
shift
./modifiedCrossSection_multiGroup $* > Outputs/${file}.out
../Utilities/diff.com modifiedCrossSection/${file}.out Benchmarks/${file}.out Outputs/${file}.out
