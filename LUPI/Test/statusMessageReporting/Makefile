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
	rm -rf Outputs/*
	./statusMessageReporting > Outputs/statusMessageReporting.0
	./statusMessageReporting 1 > Outputs/statusMessageReporting.1
	./statusMessageReporting 1 1 > Outputs/statusMessageReporting.2
	./statusMessageReporting 1 1 1 > Outputs/statusMessageReporting.3
	-diff -r Benchmarks Outputs > /dev/null; if [ $$? -ne 0 ]; then echo "FAILURE: statusMessageReporting"; fi
