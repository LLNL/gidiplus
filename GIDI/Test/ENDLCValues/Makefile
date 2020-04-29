SHELL = /bin/ksh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

.PHONY: default ENDLCValues clean realclean

PUGIXML_PATH = ../../../pugixml
PUGIXML_INCLUDE = $(PUGIXML_PATH)/include
PUGIXML_LIB = $(PUGIXML_PATH)/lib

POPs_PATH = ../../../PoPI
POPs_INCLUDE = $(POPs_PATH)/include
POPs_LIB = $(POPs_PATH)/lib

SMR_PATH = ../../../statusMessageReporting
SMR_INCLUDE = $(SMR_PATH)/include
SMR_LIB = $(SMR_PATH)/lib

NF_PATH = ../../../numericalFunctions
NF_INCLUDE = $(NF_PATH)/include
NF_LIB = $(NF_PATH)/lib

local_CXXFLAGS = $(CXXFLAGS) \
		-I../../include -L../../lib \
		-I../Utilities ../Utilities/GIDI_testUtilities.o \
		-I$(POPs_INCLUDE) -L$(POPs_LIB) \
		-I$(SMR_INCLUDE) -L$(SMR_LIB) \
		-I$(NF_INCLUDE) -L$(NF_LIB) \
		-I$(PUGIXML_INCLUDE) -L$(PUGIXML_LIB)

LIBS = -lGIDI -lPoPI -lstatusMessageReporting -lnumericalfunctions -lpugixml

default: ENDLCValues

ENDLCValues: ENDLCValues.cpp
	$(CXX) $(local_CXXFLAGS) -o ENDLCValues ENDLCValues.cpp $(LIBS)

check: ENDLCValues
	./ENDLCValues > o1
	../Utilities/diff.com ENDLCValues/ENDLCValues ENDLCValues.out o1
	./ENDLCValues 10 > o2
	../Utilities/diff.com ENDLCValues/ENDLCValues ENDLCValues_10.out o2
	./ENDLCValues 40 > o3
	../Utilities/diff.com ENDLCValues/ENDLCValues ENDLCValues_40.out o3
	./ENDLCValues 42 > o4
	../Utilities/diff.com ENDLCValues/ENDLCValues ENDLCValues_42.out o4
	./ENDLCValues 10 40 > o5
	../Utilities/diff.com ENDLCValues/ENDLCValues ENDLCValues_10_40.out o5
	./ENDLCValues 10 42 > o6
	../Utilities/diff.com ENDLCValues/ENDLCValues ENDLCValues_10_42.out o6
	./ENDLCValues 40 42 > o7
	../Utilities/diff.com ENDLCValues/ENDLCValues ENDLCValues_40_42.out o7
	./ENDLCValues 10 40 42 > o8
	../Utilities/diff.com ENDLCValues/ENDLCValues ENDLCValues_10_40_42.out o8
	./ENDLCValues 42 40 10 > o9
	../Utilities/diff.com ENDLCValues/ENDLCValues ENDLCValues_10_40_42.out o9

	./ENDLCValues --minus 5 40 --pid photon > o10
	../Utilities/diff.com ENDLCValues/ENDLCValues ENDLCValues_-5_40.out o10

clean:
	rm -f ENDLCValues o[0-9]* file1 file2

realclean: clean
	rm -rf *.dSYM
