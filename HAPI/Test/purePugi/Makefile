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

check: purePugi
	if [ ! -e Outputs ]; then mkdir Outputs; fi
	./purePugi > Outputs/purePugi.out
	../Utilities/diff.com purePugi/purePugi Benchmarks/purePugi.out Outputs/purePugi.out
