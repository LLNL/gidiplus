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
	./TNSL --tid HinCH2 > Outputs/HinCH2.out
	$(PYTHON) ../Utilities/diff.py TNSL/TNSL Benchmarks/HinCH2.out Outputs/HinCH2.out
	./TNSL --tid tnsl-Al27 > Outputs/tnsl-Al27.out
	$(PYTHON) ../Utilities/diff.py TNSL/TNSL Benchmarks/tnsl-Al27.out Outputs/tnsl-Al27.out
	./nonTNSL > Outputs/nonTNSL.out
	$(PYTHON) ../Utilities/diff.py TNSL/nonTNSL Benchmarks/nonTNSL.out Outputs/nonTNSL.out
