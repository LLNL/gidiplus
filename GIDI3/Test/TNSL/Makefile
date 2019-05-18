SHELL = /bin/ksh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

.PHONY: default TNSL clean realclean

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

local_CXXFLAGS = $(CXXFLAGS) \
		-I../../include -L../../lib \
		-I../Utilities ../Utilities/GIDI_testUtilities.o \
		-I$(POPs_INCLUDE) -L$(POPs_LIB) \
		-I$(SMR_INCLUDE) -L$(SMR_LIB) \
		-I$(NF_INCLUDE) -L$(NF_LIB) \
		-I$(PUGIXML_INCLUDE) -L$(PUGIXML_LIB)

LIBS = -lGIDI -lPoPs -lstatusMessageReporting -lnumericalfunctions -lpugixml

default: TNSL

testFiles:
	gunzip *.out.gz

TNSL: TNSL.cpp
	$(CXX) $(local_CXXFLAGS) -o TNSL TNSL.cpp $(LIBS)

check: testFiles TNSL
	./TNSL HinCH2 > o
	python ../Utilities/diff.py TNSL/TNSL HinCH2.out o
	./TNSL tnsl-Al27 > o
	python ../Utilities/diff.py TNSL/TNSL tnsl-Al27.out o

clean:
	rm -f TNSL o

realclean: clean
	rm -rf *.dSYM
