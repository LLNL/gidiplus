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
	./crossSections > Outputs/crossSections.out
	../Utilities/diff.com crossSections/crossSections Benchmarks/crossSections.out Outputs/crossSections.out
	./crossSections --map ../Data/MG_MC/all_maps.map --tid Li6 --pid H1 > Outputs/crossSections_H1+Li6.out
	../Utilities/diff.com crossSections/crossSections Benchmarks/crossSections_H1+Li6.out Outputs/crossSections_H1+Li6.out
	./crossSections --map ../Data/MG_MC/all_maps.map --tid Li6 --pid H1 --noCoulomb > Outputs/crossSections_H1+Li6_noCoulomb.out
	../Utilities/diff.com crossSections/crossSections Benchmarks/crossSections_H1+Li6_noCoulomb.out Outputs/crossSections_H1+Li6_noCoulomb.out
