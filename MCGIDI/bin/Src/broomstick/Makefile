SHELL = /bin/sh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

OpenMPFlags = -fopenmp

ifdef GIDI_PLUS_INCLUDE_PYTHON_BUILDING
	LIBS_extras = -lGIDIP_python $(shell python3-config --libs)
else
	LIBS_extras = -lGIDIP_python
endif

GIDI_PLUS_PATH ?= $(abspath ../../../..)
CppSources = $(sort $(wildcard *.cpp))

include $(GIDI_PLUS_PATH)/Makefile.paths
include ../Makefile.bin
