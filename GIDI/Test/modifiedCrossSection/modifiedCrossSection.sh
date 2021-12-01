#! /usr/bin/env bash

file=$1
shift
./modifiedCrossSection $* > Outputs/$file
../Utilities/diff.com modifiedCrossSection/$file Benchmarks/$file Outputs/$file
