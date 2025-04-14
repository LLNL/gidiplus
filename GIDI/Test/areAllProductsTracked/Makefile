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
	./areAllProductsTracked --tid O16 				> Outputs/areAllProductsTracked.n+O16.out
	../Utilities/diff.com areAllProductsTracked/areAllProductsTracked Benchmarks/areAllProductsTracked.n+O16.out Outputs/areAllProductsTracked.n+O16.out
	./areAllProductsTracked --pid H1 --tid Li6 		> Outputs/areAllProductsTracked.H1+Li6.out
	../Utilities/diff.com areAllProductsTracked/areAllProductsTracked Benchmarks/areAllProductsTracked.H1+Li6.out Outputs/areAllProductsTracked.H1+Li6.out
	./areAllProductsTracked --pid photon --tid O16 	> Outputs/areAllProductsTracked.photon+O16.out
	../Utilities/diff.com areAllProductsTracked/areAllProductsTracked Benchmarks/areAllProductsTracked.photon+O16.out Outputs/areAllProductsTracked.photon+O16.out
