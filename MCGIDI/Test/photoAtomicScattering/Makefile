SHELL = /bin/ksh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

.PHONY: default sampleCoherent sampleIncoherent POfMu clean realclean

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

GIDI_PATH = ../../../GIDI
GIDI_INCLUDE = $(GIDI_PATH)/include
GIDI_LIB = $(GIDI_PATH)/lib


local_CXXFLAGS = $(CXXFLAGS) -I../../include -L../../lib -I../Utilities \
		-I$(GIDI_INCLUDE) -L$(GIDI_LIB) \
		-I$(POPs_INCLUDE) -L$(POPs_LIB) \
		-I$(SMR_INCLUDE) -L$(SMR_LIB) \
		-I$(NF_INCLUDE) -L$(NF_LIB) \
        -I$(PUGIXML_INCLUDE) -L$(PUGIXML_LIB) \
		-I../Utilities -L../Utilities

LIBS = -lMCGIDI -lGIDI -lPoPI -lstatusMessageReporting -lnumericalfunctions -lMCGIDI_testUtilities -lpugixml

default: sampleCoherent sampleIncoherent POfMu

sampleCoherent: sampleCoherent.cpp
	$(CXX) $(local_CXXFLAGS) -o sampleCoherent sampleCoherent.cpp $(LIBS)

sampleIncoherent: sampleIncoherent.cpp
	$(CXX) $(local_CXXFLAGS) -o sampleIncoherent sampleIncoherent.cpp $(LIBS)

POfMu: POfMu.cpp
	$(CXX) $(local_CXXFLAGS) -o POfMu POfMu.cpp $(LIBS)

check: sampleCoherent
	./sampleCoherent > o
#	../Utilities/diff.com sampleCoherent/sampleCoherent sampleCoherent.out o

clean:
	rm -f POfMu sampleCoherent sampleIncoherent o

realclean: clean
	rm -rf *.dSYM
