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

check: domainHash
	if [ ! -e Outputs ]; then mkdir Outputs; fi
	-./domainHash > Outputs/domainHash.out; if [ $$? != 0 ]; then echo "domainHash.cpp failed with errors"; fi
	diff -q Benchmarks/domainHash.out Outputs/domainHash.out > /dev/null; if [ $$? != 0 ]; then echo "domainHash.cpp failed"; fi
