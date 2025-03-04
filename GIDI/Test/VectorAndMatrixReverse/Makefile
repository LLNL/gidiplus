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
	./vectorReverse > Outputs/vectorReverse.out
	../Utilities/diff.com vectorReverse/vectorReverse Benchmarks/vectorReverse.out Outputs/vectorReverse.out
	./matrixReverse > Outputs/matrixReverse.out
	../Utilities/diff.com matrixReverse/matrixReverse Benchmarks/matrixReverse.out Outputs/matrixReverse.out
