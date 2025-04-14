SHELL = /bin/sh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

# Should build with "make CXXFLAGS=-std=c++11"

GIDI_PLUS_PATH ?= $(abspath ../..)
CppSources := $(wildcard *.cpp)

include $(GIDI_PLUS_PATH)/Makefile.paths

.PHONY: default clean realclean

local_CXXFLAGS = $(CXXFLAGS) \
		-I$(GIDI_PATH)/include -L$(GIDI_PATH)/lib \
		-I$(GIDI_PATH)/Test/Utilities -L$(GIDI_PATH)/Test/Utilities \
		-I$(POPI_INCLUDE) -L$(POPI_LIB) \
		-I$(SMR_INCLUDE) -L$(SMR_LIB) \
		-I$(NF_INCLUDE) -L$(NF_LIB) \
		-I$(PUGIXML_INCLUDE) -L$(PUGIXML_LIB)

LIBS = -lGIDI_testUtilities -lGIDI -lPoPI -lstatusMessageReporting -lnumericalfunctions -lpugixml

Executables := $(CppSources:%.cpp=%)

default: $(Executables)

%: %.cpp
	$(CXX) $(local_CXXFLAGS) $< -o $@ $(LIBS)

check: 

clean:
	rm -f $(Executables)

realclean: clean
	rm -rf *.dSYM
