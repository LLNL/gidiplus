SHELL = /bin/ksh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

.PHONY: default sampleTerrellPromptNeutronDistribution clean realclean

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


local_CXXFLAGS = $(CXXFLAGS) -I../../include -L../../lib \
		-I$(GIDI_INCLUDE) -L$(GIDI_LIB) \
		-I$(POPs_INCLUDE) -L$(POPs_LIB) \
		-I$(SMR_INCLUDE) -L$(SMR_LIB) \
		-I$(NF_INCLUDE) -L$(NF_LIB) \
		-I$(PUGIXML_INCLUDE) -L$(PUGIXML_LIB) \
		-I../Utilities -L../Utilities

LIBS = -lMCGIDI -lGIDI -lPoPI -lstatusMessageReporting -lnumericalfunctions -lMCGIDI_testUtilities -lpugixml

default: sampleTerrellPromptNeutronDistribution

sampleTerrellPromptNeutronDistribution: sampleTerrellPromptNeutronDistribution.cpp
	$(CXX) $(local_CXXFLAGS) -o sampleTerrellPromptNeutronDistribution sampleTerrellPromptNeutronDistribution.cpp $(LIBS)

check: sampleTerrellPromptNeutronDistribution
	./sampleTerrellPromptNeutronDistribution > o
	../Utilities/diff.com sampleTerrellPromptNeutronDistribution/sampleTerrellPromptNeutronDistribution sampleTerrellPromptNeutronDistribution.out o
	./sampleTerrellPromptNeutronDistribution -t > o
	../Utilities/diff.com sampleTerrellPromptNeutronDistribution/sampleTerrellPromptNeutronDistribution sampleTerrellPromptNeutronDistribution_-t.out o

clean:
	rm -f sampleTerrellPromptNeutronDistribution o

realclean: clean
