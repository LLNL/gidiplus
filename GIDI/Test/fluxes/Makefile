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
	./fluxes > Outputs/fluxes.out
	diff Benchmarks/fluxes.out Outputs/fluxes.out > /dev/null; if [ $$? -ne 0 ]; then echo "FAILURE: fluxes/fluxes.cpp: difference in output"; fi
	./bdfls3dFlux > Outputs/bdfls3dFlux.out
	diff Benchmarks/bdfls3dFlux.out Outputs/bdfls3dFlux.out > /dev/null; if [ $$? -ne 0 ]; then echo "FAILURE: bdfls3dFlux/bdfls3dFlux.cpp: difference in output"; fi
