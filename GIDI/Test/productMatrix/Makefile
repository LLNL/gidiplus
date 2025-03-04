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
	./productMatrix --tid O16 > Outputs/O16.out
	$(PYTHON) ../Utilities/diff.py "productMatrix/productMatrix O16" Benchmarks/O16.out Outputs/O16.out
	./productMatrix --tid Th227 > Outputs/Th227.noDelayed.out
	$(PYTHON) ../Utilities/diff.py "productMatrix/productMatrix Th227" Benchmarks/Th227.noDelayed.out Outputs/Th227.noDelayed.out
	./productMatrix --tid Th227 --delayed > Outputs/Th227.delayed.out
	$(PYTHON) ../Utilities/diff.py "productMatrix/productMatrix Th227" Benchmarks/Th227.delayed.out Outputs/Th227.delayed.out
