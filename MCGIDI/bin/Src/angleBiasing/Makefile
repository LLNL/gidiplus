SHELL = /bin/ksh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

.PHONY: default bin angleBiasing clean realclean

ROOT_PATH = ../../../..

PUGIXML_PATH = $(ROOT_PATH)/pugixml
PUGIXML_INCLUDE = $(PUGIXML_PATH)/include
PUGIXML_LIB = $(PUGIXML_PATH)/lib

POPs_PATH = $(ROOT_PATH)/PoPI
POPs_INCLUDE = $(POPs_PATH)/include
POPs_LIB = $(POPs_PATH)/lib

SMR_PATH = $(ROOT_PATH)/statusMessageReporting
SMR_INCLUDE = $(SMR_PATH)/include
SMR_LIB = $(SMR_PATH)/lib

NF_PATH = $(ROOT_PATH)/numericalFunctions
NF_INCLUDE = $(NF_PATH)/include
NF_LIB = $(NF_PATH)/lib

GIDI_PATH = $(ROOT_PATH)/GIDI
GIDI_INCLUDE = $(GIDI_PATH)/include
GIDI_LIB = $(GIDI_PATH)/lib

GIDI_UTILITIES_PATH = $(GIDI_PATH)/Test/Utilities

local_CXXFLAGS = $(CXXFLAGS) -I../../../include -L../../../lib -I../../../Test/Utilities -L../../../Test/Utilities \
		-I$(GIDI_PATH)/Test/Utilities -L$(GIDI_PATH)/Test/Utilities \
		-I$(GIDI_INCLUDE) -L$(GIDI_LIB) \
		-I$(POPs_INCLUDE) -L$(POPs_LIB) \
		-I$(SMR_INCLUDE) -L$(SMR_LIB) \
		-I$(NF_INCLUDE) -L$(NF_LIB) \
		-I$(PUGIXML_INCLUDE) -L$(PUGIXML_LIB)

LIBS = -lGIDI_testUtilities -lMCGIDI_testUtilities -lMCGIDI -lGIDI -lPoPI -lstatusMessageReporting -lnumericalfunctions -lpugixml

default: angleBiasing

bin: angleBiasing
	mv angleBiasing ../..

angleBiasing: angleBiasing.cpp
	$(CXX) $(local_CXXFLAGS) -o angleBiasing angleBiasing.cpp $(LIBS)

clean:
	rm -f angleBiasing

realclean: clean
