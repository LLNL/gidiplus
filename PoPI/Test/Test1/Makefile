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
	./test1 > Outputs/test1.out
	diff -q Benchmarks/test1.out Outputs/test1.out > /dev/null; if [ $$? != 0 ]; then echo "test1.cpp failed"; fi
	./test2 > Outputs/test2.out
	diff -q Benchmarks/test2.out Outputs/test2.out > /dev/null; if [ $$? != 0 ]; then echo "test2.cpp failed"; fi
	./test3 > Outputs/test3.out
	diff -q Benchmarks/test3.out Outputs/test3.out > /dev/null; if [ $$? != 0 ]; then echo "test3.cpp failed"; fi
