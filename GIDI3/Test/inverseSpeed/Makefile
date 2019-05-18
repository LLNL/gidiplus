SHELL = /bin/ksh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

.PHONY: default inverseSpeed clean realclean

PUGIXML_PATH = ../../../pugixml
PUGIXML_INCLUDE = $(PUGIXML_PATH)/include
PUGIXML_LIB = $(PUGIXML_PATH)/lib

POPs_PATH = ../../../PoPsCpp
POPs_INCLUDE = $(POPs_PATH)/include
POPs_LIB = $(POPs_PATH)/lib

SMR_PATH = ../../../statusMessageReporting
SMR_INCLUDE = $(SMR_PATH)/include
SMR_LIB = $(SMR_PATH)/lib

NF_PATH = ../../../numericalFunctions
NF_INCLUDE = $(NF_PATH)/include
NF_LIB = $(NF_PATH)/lib

local_CXXFLAGS = $(CXXFLAGS) -I../../include -L../../lib \
		-I$(POPs_INCLUDE) -L$(POPs_LIB) \
		-I$(SMR_INCLUDE) -L$(SMR_LIB) \
		-I$(NF_INCLUDE) -L$(NF_LIB) \
		-I$(PUGIXML_INCLUDE) -L$(PUGIXML_LIB) \
		-I../Utilities -L../Utilities

LIBS = -lGIDI -lPoPs -lstatusMessageReporting -lnumericalfunctions -lGIDI_testUtilities -lpugixml

default: inverseSpeed

inverseSpeed: inverseSpeed.cpp
	$(CXX) $(local_CXXFLAGS) -o inverseSpeed inverseSpeed.cpp $(LIBS)

check: inverseSpeed
	./inverseSpeed > o
	../Utilities/diff.com inverseSpeed/inverseSpeed inverseSpeed.out o

clean:
	rm -f inverseSpeed o file1 file2

realclean: clean
	rm -rf *.dSYM
