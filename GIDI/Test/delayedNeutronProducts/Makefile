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
	./delayedNeutronProducts --tid O16 > Outputs/delayedNeutronProducts.O16.out
	python ../Utilities/diff.py "delayedNeutronProducts/delayedNeutronProducts O16" Benchmarks/delayedNeutronProducts.O16.out Outputs/delayedNeutronProducts.O16.out
	./delayedNeutronProducts --tid Th227 > Outputs/delayedNeutronProducts.Th227.out
	python ../Utilities/diff.py "delayedNeutronProducts/delayedNeutronProducts Th227" Benchmarks/delayedNeutronProducts.Th227.out Outputs/delayedNeutronProducts.Th227.out
