SHELL = /bin/ksh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

.PHONY: default upscatterModelA sampleTargetBetaForUpscatterModelA sampleBetaFromMaxwellian upscatterModelASingle sampleBetaFromMaxwellian_MCAPM clean realclean

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
		-I../Utilities

LIBS = -lMCGIDI -lGIDI -lPoPI -lstatusMessageReporting -lnumericalfunctions -lpugixml

default: upscatterModelA sampleTargetBetaForUpscatterModelA sampleBetaFromMaxwellian upscatterModelASingle sampleBetaFromMaxwellian_MCAPM

upscatterModelA: upscatterModelA.cpp
	$(CXX) $(local_CXXFLAGS) -o upscatterModelA upscatterModelA.cpp $(LIBS)

sampleTargetBetaForUpscatterModelA: sampleTargetBetaForUpscatterModelA.cpp
	$(CXX) $(local_CXXFLAGS) -o sampleTargetBetaForUpscatterModelA sampleTargetBetaForUpscatterModelA.cpp $(LIBS)

sampleBetaFromMaxwellian: sampleBetaFromMaxwellian.cpp
	$(CXX) $(local_CXXFLAGS) -o sampleBetaFromMaxwellian sampleBetaFromMaxwellian.cpp $(LIBS)

upscatterModelASingle: upscatterModelASingle.cpp
	$(CXX) $(local_CXXFLAGS) -o upscatterModelASingle upscatterModelASingle.cpp $(LIBS)

sampleBetaFromMaxwellian_MCAPM: sampleBetaFromMaxwellian_MCAPM.cpp
	$(CXX) $(local_CXXFLAGS) -o sampleBetaFromMaxwellian_MCAPM sampleBetaFromMaxwellian_MCAPM.cpp $(LIBS)

check: upscatterModelA
	./upscatterModelA > o
#	../Utilities/diff.com upscatterModelA/upscatterModelA upscatterModelA.out o

clean:
	rm -f upscatterModelA sampleTargetBetaForUpscatterModelA sampleBetaFromMaxwellian upscatterModelASingle sampleBetaFromMaxwellian_MCAPM *.dat

realclean: clean
