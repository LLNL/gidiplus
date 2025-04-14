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
	./gain --tid O16 > Outputs/O16.n.out
	$(PYTHON) ../Utilities/diff.py "gain/gain O16" Benchmarks/O16.n.out Outputs/O16.n.out
	./gain --tid Th227 > Outputs/Th227.n.out
	$(PYTHON) ../Utilities/diff.py "gain/gain Th227" Benchmarks/Th227.n.out Outputs/Th227.n.out
