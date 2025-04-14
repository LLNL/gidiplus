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

check: parseXML
	if [ ! -e Outputs ]; then mkdir Outputs; fi
	./parseXML > Outputs/parseXML.out
	../Utilities/diff.com parseXML/parseXML ../purePugi/Benchmarks/purePugi.out Outputs/parseXML.out
