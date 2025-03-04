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
	./test3_1 > Outputs/test3_1.out
	diff -q Benchmarks/test3_1.out Outputs/test3_1.out > /dev/null; if [ $$? != 0 ]; then echo "test3_1.cpp failed"; fi
	diff -q  Benchmarks/pops3.xml.out Outputs/pops3.xml.out > /dev/null; if [ $$? != 0 ]; then echo "pops3.xml.out failed"; fi
