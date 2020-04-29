SHELL = /bin/ksh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

.PHONY: default crossSections crossSectionSum clean realclean

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

default: crossSections crossSectionSum

crossSections: crossSections.cpp
	$(CXX) $(local_CXXFLAGS) -o crossSections crossSections.cpp $(LIBS)

crossSectionSum: crossSectionSum.cpp
	$(CXX) $(local_CXXFLAGS) -o crossSectionSum crossSectionSum.cpp $(LIBS)

check: crossSections crossSectionSum
	./crossSections > o
	../Utilities/diff.com crossSections/crossSections crossSections.out o

	./crossSections --pid photon --tid O16 --map ../../../GIDI/Test/Data/MG_MC/all.map > o
	../Utilities/diff.com crossSections/crossSections-1 crossSections.photon+O16.out o
	./crossSections --pid photon --tid O16 --map ../../../GIDI/Test/Data/MG_MC/all.map -n > o
	../Utilities/diff.com crossSections/crossSections-2 crossSections.photon+O16.out o
	./crossSections --pid photon --tid O16 --map ../../../GIDI/Test/Data/MG_MC/all.map -a > o
	../Utilities/diff.com crossSections/crossSections-3 crossSections.photon+O16.atomic.out o
	./crossSections --pid photon --tid O16 --map ../../../GIDI/Test/Data/MG_MC/all.map -a -n > o
	../Utilities/diff.com crossSections/crossSections-4 crossSections.photon+O16.atomic+nuclear.out o

	./crossSections --tid HinCH2 --map ../../../GIDI/Test/Data/MG_MC/neutrons/all.map > o
	../Utilities/diff.com crossSections/crossSections-HinCH2 crossSections.HinCH2.out o

	./crossSectionSum > o
	../Utilities/diff.com crossSectionSum/crossSectionSum crossSectionSum.out o

clean:
	rm -f crossSections crossSectionSum o

realclean: clean
