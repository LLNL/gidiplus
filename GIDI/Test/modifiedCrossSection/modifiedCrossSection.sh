#! /usr/bin/env bash

file=$1
shift
./modifiedCrossSection $* > Outputs/${file}.out
../Utilities/diff.com modifiedCrossSection/${file}.out Benchmarks/${file}.out Outputs/${file}.out

./modifiedCrossSection_nullptr $* > Outputs/${file}_nullptr.out
python3 diff_nullptr.py modifiedCrossSection/${file}_nullptr.out Outputs/${file}.out Outputs/${file}_nullptr.out
