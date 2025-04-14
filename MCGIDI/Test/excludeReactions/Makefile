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

check: excludeReactions
	if [ ! -e Outputs ]; then mkdir Outputs; fi
	./excludeReactions > Outputs/excludeReactions.out
	../Utilities/diff.com excludeReactions/excludeReactions Benchmarks/excludeReactions.out Outputs/excludeReactions.out
	./excludeReactions 2 > Outputs/excludeReactions_2.out
	../Utilities/diff.com excludeReactions/excludeReactions Benchmarks/excludeReactions_2.out Outputs/excludeReactions_2.out
	./excludeReactions 5 > Outputs/excludeReactions_5.out
	../Utilities/diff.com excludeReactions/excludeReactions Benchmarks/excludeReactions_5.out Outputs/excludeReactions_5.out
	./excludeReactions 0 2 5 > Outputs/excludeReactions_0_2_5.out
	../Utilities/diff.com excludeReactions/excludeReactions Benchmarks/excludeReactions_0_2_5.out Outputs/excludeReactions_0_2_5.out

	./excludeReactions --pid n --tid O16 > Outputs/excludeReactions_n_O16.out
	../Utilities/diff.com excludeReactions/excludeReactions Benchmarks/excludeReactions_n_O16.out Outputs/excludeReactions_n_O16.out
	./excludeReactions --pid n --tid O16 3 4 5 6 7 8 > Outputs/excludeReactions_n_O16_3-8.out
	../Utilities/diff.com excludeReactions/excludeReactions Benchmarks/excludeReactions_n_O16_3-8.out Outputs/excludeReactions_n_O16_3-8.out
	./excludeReactions --pid n --tid O16 3 4 5 6 7 8 17 18 19 20 21 22 23 24 > Outputs/excludeReactions_n_O16_3-8_17-24.out
	../Utilities/diff.com excludeReactions/excludeReactions Benchmarks/excludeReactions_n_O16_3-8_17-24.out Outputs/excludeReactions_n_O16_3-8_17-24.out

	./excludeReactions --pid n --tid tnsl-Al27 --map ../../../GIDI/Test/Data/MG_MC/neutrons/all.map > Outputs/excludeReactions_tnsl-Al27.out
	../Utilities/diff.com excludeReactions/excludeReactions Benchmarks/excludeReactions_tnsl-Al27.out Outputs/excludeReactions_tnsl-Al27.out
