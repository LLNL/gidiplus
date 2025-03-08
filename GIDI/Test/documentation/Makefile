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
	./documentation --tid O16 > Outputs/O16.out
	../Utilities/diff.com documentation/documentation Benchmarks/O16.out Outputs/O16.out
	./documentation --tid Th227 > Outputs/Th227.out
	../Utilities/diff.com documentation/documentation Benchmarks/Th227.out Outputs/Th227.out
