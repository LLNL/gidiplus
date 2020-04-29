SHELL = /bin/ksh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

# Should build with "make CXXFLAGS=-std=c++11"

.PHONY: default clean realclean

SRCS := $(wildcard *.cpp)
BINS := $(SRCS:%.cpp=%)

PUGIXML_PATH = ../../pugixml
PUGIXML_INCLUDE = $(PUGIXML_PATH)/include
PUGIXML_LIB = $(PUGIXML_PATH)/lib

POPs_PATH = ../../PoPI
POPs_INCLUDE = $(POPs_PATH)/include
POPs_LIB = $(POPs_PATH)/lib

SMR_PATH = ../../statusMessageReporting
SMR_INCLUDE = $(SMR_PATH)/include
SMR_LIB = $(SMR_PATH)/lib

NF_PATH = ../../numericalFunctions
NF_INCLUDE = $(NF_PATH)/include
NF_LIB = $(NF_PATH)/lib

local_CXXFLAGS = $(CXXFLAGS) \
		-I../include -L../lib \
		-I../Test/Utilities -L../Test/Utilities \
		-I$(POPs_INCLUDE) -L$(POPs_LIB) \
		-I$(SMR_INCLUDE) -L$(SMR_LIB) \
		-I$(NF_INCLUDE) -L$(NF_LIB) \
		-I$(PUGIXML_INCLUDE) -L$(PUGIXML_LIB)

LIBS = -lGIDI_testUtilities -lGIDI -lPoPI -lstatusMessageReporting -lnumericalfunctions -lpugixml

default: $(BINS)

%: %.cpp
	$(CXX) $(local_CXXFLAGS) $(LIBS) $< -o $@

check: 

clean:
	rm -f $(BINS)

realclean: clean
	rm -rf *.dSYM
