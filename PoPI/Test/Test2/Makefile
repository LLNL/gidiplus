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
	./test2_1 > Outputs/test2_1.out
	diff -q Benchmarks/test2_1.out Outputs/test2_1.out > /dev/null; if [ $$? != 0 ]; then echo "test2_1.cpp failed"; fi
	./test2_2 > Outputs/test2_2.out
	diff -q Benchmarks/test2_2.out Outputs/test2_2.out > /dev/null; if [ $$? != 0 ]; then echo "test2_2.cpp failed"; fi
