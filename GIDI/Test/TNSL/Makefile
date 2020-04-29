SHELL = /bin/ksh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

.PHONY: default TNSL nonTNSL clean realclean

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
		-I../Utilities -L../Utilities \
		-I$(POPs_INCLUDE) -L$(POPs_LIB) \
		-I$(SMR_INCLUDE) -L$(SMR_LIB) \
		-I$(NF_INCLUDE) -L$(NF_LIB) \
		-I$(PUGIXML_INCLUDE) -L$(PUGIXML_LIB)

LIBS = -lGIDI_testUtilities -lGIDI -lPoPI -lstatusMessageReporting -lnumericalfunctions -lpugixml

default: TNSL nonTNSL

TNSL: TNSL.cpp
	$(CXX) $(local_CXXFLAGS) -o TNSL TNSL.cpp $(LIBS)

nonTNSL: nonTNSL.cpp
	$(CXX) $(local_CXXFLAGS) -o nonTNSL nonTNSL.cpp $(LIBS)

check: TNSL nonTNSL
	./TNSL --tid HinCH2 > o1
	python ../Utilities/diff.py TNSL/TNSL HinCH2.out o1
	./TNSL --tid tnsl-Al27 > o2
	python ../Utilities/diff.py TNSL/TNSL tnsl-Al27.out o2
	./nonTNSL > o3
	python ../Utilities/diff.py TNSL/nonTNSL nonTNSL.out o3

clean:
	rm -f TNSL nonTNSL o[0-9]*

realclean: clean
	rm -rf *.dSYM
