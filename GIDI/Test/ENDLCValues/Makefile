SHELL = /bin/sh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

GIDI_PLUS_PATH ?= $(abspath ../../..)
CppSource = $(sort $(wildcard *.cpp))

include $(GIDI_PLUS_PATH)/Makefile.paths
include ../Makefile.check

check: $(Executables)
	if [ ! -e Outputs ]; then mkdir Outputs; fi
	./ENDLCValues > Outputs/ENDLCValues.out
	../Utilities/diff.com ENDLCValues/ENDLCValues Benchmarks/ENDLCValues.out Outputs/ENDLCValues.out
	./ENDLCValues 10 > Outputs/ENDLCValues_10.out
	../Utilities/diff.com ENDLCValues/ENDLCValues Benchmarks/ENDLCValues_10.out Outputs/ENDLCValues_10.out
	./ENDLCValues 40 > Outputs/ENDLCValues_40.out
	../Utilities/diff.com ENDLCValues/ENDLCValues Benchmarks/ENDLCValues_40.out Outputs/ENDLCValues_40.out
	./ENDLCValues 42 > Outputs/ENDLCValues_42.out
	../Utilities/diff.com ENDLCValues/ENDLCValues Benchmarks/ENDLCValues_42.out Outputs/ENDLCValues_42.out
	./ENDLCValues 10 40 > Outputs/ENDLCValues_10_40.out
	../Utilities/diff.com ENDLCValues/ENDLCValues Benchmarks/ENDLCValues_10_40.out Outputs/ENDLCValues_10_40.out
	./ENDLCValues 10 42 > Outputs/ENDLCValues_10_42.out
	../Utilities/diff.com ENDLCValues/ENDLCValues Benchmarks/ENDLCValues_10_42.out Outputs/ENDLCValues_10_42.out
	./ENDLCValues 40 42 > Outputs/ENDLCValues_40_42.out
	../Utilities/diff.com ENDLCValues/ENDLCValues Benchmarks/ENDLCValues_40_42.out Outputs/ENDLCValues_40_42.out
	./ENDLCValues 10 40 42 > Outputs/ENDLCValues_10_40_42.out
	../Utilities/diff.com ENDLCValues/ENDLCValues Benchmarks/ENDLCValues_10_40_42.out Outputs/ENDLCValues_10_40_42.out
	./ENDLCValues 42 40 10 > Outputs/ENDLCValues_10_40_42.out
	../Utilities/diff.com ENDLCValues/ENDLCValues Benchmarks/ENDLCValues_10_40_42.out Outputs/ENDLCValues_10_40_42.out

	./ENDLCValues --minus 5 40 --pid photon > Outputs/ENDLCValues_-5_40.out
	../Utilities/diff.com ENDLCValues/ENDLCValues Benchmarks/ENDLCValues_-5_40.out Outputs/ENDLCValues_-5_40.out
