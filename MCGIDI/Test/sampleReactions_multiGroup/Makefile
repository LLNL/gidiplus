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

check: sampleReactions_multiGroup
	if [ ! -e Outputs ]; then mkdir Outputs; fi
	./sampleReactions_multiGroup -n -10 > Outputs/sampleReactions_multiGroup.out
	../Utilities/diff.com sampleReactions_multiGroup/sampleReactions_multiGroup Benchmarks/sampleReactions_multiGroup.out Outputs/sampleReactions_multiGroup.out
	./sampleReactions_multiGroup --pid photon --tid O16 --map ../../../GIDI/Test/Data/MG_MC/all.map > Outputs/sampleReactions_multiGroup.photon+O16.out
	../Utilities/diff.com sampleReactions_multiGroup/sampleReactions_multiGroup-1 Benchmarks/sampleReactions_multiGroup.photon+O16.out Outputs/sampleReactions_multiGroup.photon+O16.out
	./sampleReactions_multiGroup --pid photon --tid O16 --map ../../../GIDI/Test/Data/MG_MC/all.map -pn > Outputs/sampleReactions_multiGroup.photon+O16.out
	../Utilities/diff.com sampleReactions_multiGroup/sampleReactions_multiGroup-2 Benchmarks/sampleReactions_multiGroup.photon+O16.out Outputs/sampleReactions_multiGroup.photon+O16.out
	./sampleReactions_multiGroup --pid photon --tid O16 --map ../../../GIDI/Test/Data/MG_MC/all.map -pa > Outputs/sampleReactions_multiGroup.photon+O16.atomic.out
	../Utilities/diff.com sampleReactions_multiGroup/sampleReactions_multiGroup-3 Benchmarks/sampleReactions_multiGroup.photon+O16.atomic.out Outputs/sampleReactions_multiGroup.photon+O16.atomic.out
	./sampleReactions_multiGroup --pid photon --tid O16 --map ../../../GIDI/Test/Data/MG_MC/all.map -pa -pn > Outputs/sampleReactions_multiGroup.photon+O16.atomic+nuclear.out
	../Utilities/diff.com sampleReactions_multiGroup/sampleReactions_multiGroup-4 Benchmarks/sampleReactions_multiGroup.photon+O16.atomic+nuclear.out Outputs/sampleReactions_multiGroup.photon+O16.atomic+nuclear.out
