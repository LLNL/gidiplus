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
	./aliasesUnreferenced > Outputs/aliasesUnreferenced.out
	diff -q Benchmarks/aliasesUnreferenced.out Outputs/aliasesUnreferenced.out > /dev/null; if [ $$? != 0 ]; then echo "aliasesUnreferenced.cpp failed"; fi
