SHELL = /bin/ksh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

.PHONY: default gain clean realclean

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
        -I$(PUGIXML_INCLUDE) -L$(PUGIXML_LIB)

LIBS = -lGIDI -lPoPs -lstatusMessageReporting -lnumericalfunctions -lpugixml

default: gain

gain: gain.cpp
	$(CXX) $(local_CXXFLAGS) -o gain gain.cpp $(LIBS)

check: gain
	./gain O16 > o
	python ../Utilities/diff.py "gain/gain O16" O16.n.out o
	./gain Th227 > o
	python ../Utilities/diff.py "gain/gain Th227" Th227.n.out o

clean:
	rm -f gain o file1 file2

realclean: clean
	rm -rf *.dSYM
