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
	./multiplicity --tid O16 > Outputs/O16.out
	../Utilities/diff.com multiplicity/multiplicity Benchmarks/O16.out Outputs/O16.out
	./multiplicity --tid Th227 > Outputs/Th227.out
	../Utilities/diff.com multiplicity/multiplicity Benchmarks/Th227.out Outputs/Th227.out
