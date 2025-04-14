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

check: crossSection_multiGroup crossSectionSum_multiGroup
	if [ ! -e Outputs ]; then mkdir Outputs; fi
	./crossSection_multiGroup > Outputs/crossSection_multiGroup.out
	../Utilities/diff.com crossSection_multiGroup/crossSection_multiGroup Benchmarks/crossSection_multiGroup.out Outputs/crossSection_multiGroup.out

	./crossSection_multiGroup --pid photon --tid O16 --map ../../../GIDI/Test/Data/MG_MC/all.map > Outputs/crossSection_multiGroup.photon+O16.out
	../Utilities/diff.com crossSection_multiGroup/crossSection_multiGroup-1 Benchmarks/crossSection_multiGroup.photon+O16.out Outputs/crossSection_multiGroup.photon+O16.out
	./crossSection_multiGroup --pid photon --tid O16 --map ../../../GIDI/Test/Data/MG_MC/all.map -n > Outputs/crossSection_multiGroup.photon+O16.nuclear.out
	../Utilities/diff.com crossSection_multiGroup/crossSection_multiGroup-2 Benchmarks/crossSection_multiGroup.photon+O16.nuclear.out Outputs/crossSection_multiGroup.photon+O16.nuclear.out
	./crossSection_multiGroup --pid photon --tid O16 --map ../../../GIDI/Test/Data/MG_MC/all.map -a > Outputs/crossSection_multiGroup.photon+O16.atomic.out
	../Utilities/diff.com crossSection_multiGroup/crossSection_multiGroup-3 Benchmarks/crossSection_multiGroup.photon+O16.atomic.out Outputs/crossSection_multiGroup.photon+O16.atomic.out
	./crossSection_multiGroup --pid photon --tid O16 --map ../../../GIDI/Test/Data/MG_MC/all.map -a -n > Outputs/crossSection_multiGroup.photon+O16.atomic+nuclear.out
	../Utilities/diff.com crossSection_multiGroup/crossSection_multiGroup-4 Benchmarks/crossSection_multiGroup.photon+O16.atomic+nuclear.out Outputs/crossSection_multiGroup.photon+O16.atomic+nuclear.out

	./crossSectionSum_multiGroup > Outputs/crossSectionSum_multiGroup.out
	../Utilities/diff.com crossSectionSum_multiGroup/crossSectionSum_multiGroup Benchmarks/crossSectionSum_multiGroup.out Outputs/crossSectionSum_multiGroup.out
