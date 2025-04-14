SHELL = /bin/sh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

COMPILE_UTILITIES = 1
GIDI_PLUS_PATH ?= $(abspath ../../..)

CppSources = $(sort $(wildcard *.cpp))
CppObjs = $(CppSources:.cpp=.o)
Archive = libGIDI_testUtilities.a

include $(GIDI_PLUS_PATH)/Makefile.paths
include ../Makefile.check

default:
	echo ""
	echo "  -- GIDI/Test/Utilities --"
	$(MAKE) $(Archive)

%.o: %.cpp
	echo "    "$<
	$(CXX) -c $(local_CXXFLAGS) $< -o $@

$(Archive): $(CppObjs)
	rm -f $@
	$(AR) -cr $@ $(CppObjs)

GIDI_testUtilities.cpp: GIDI_testUtilities.hpp

check:
