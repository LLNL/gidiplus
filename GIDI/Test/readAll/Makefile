SHELL = /bin/ksh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

GIDI_PLUS_PATH ?= $(abspath ../../..)
CppSource = $(sort $(wildcard *.cpp))

include ../../Makefile.paths
include ../Makefile.check

check: $(Executables)
	if [ ! -e Outputs ]; then mkdir Outputs; fi
	./readAll > Outputs/all.out
	../Utilities/diff.com readAll/readAll Benchmarks/all.out Outputs/all.out
	./readAll --map "../Data/MG_MC/all_maps.map" > Outputs/all_maps.out
	../Utilities/diff.com readAll/readAll Benchmarks/all_maps.out Outputs/all_maps.out
