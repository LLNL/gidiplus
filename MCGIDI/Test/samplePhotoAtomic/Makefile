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

check: samplePhotoAtomic
	if [ ! -e Outputs ]; then mkdir Outputs; fi
	./samplePhotoAtomic > Outputs/samplePhotoAtomic.out
	../Utilities/diff.com samplePhotoAtomic/samplePhotoAtomic Benchmarks/samplePhotoAtomic.out Outputs/samplePhotoAtomic.out
