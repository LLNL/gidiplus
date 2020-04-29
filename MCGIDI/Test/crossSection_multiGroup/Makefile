SHELL = /bin/ksh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

.PHONY: default crossSection_multiGroup crossSectionSum_multiGroup clean realclean

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

default: crossSection_multiGroup crossSectionSum_multiGroup

crossSection_multiGroup: crossSection_multiGroup.cpp
	$(CXX) $(local_CXXFLAGS) -o crossSection_multiGroup crossSection_multiGroup.cpp $(LIBS)

crossSectionSum_multiGroup: crossSectionSum_multiGroup.cpp
	$(CXX) $(local_CXXFLAGS) -o crossSectionSum_multiGroup crossSectionSum_multiGroup.cpp $(LIBS)

check: crossSection_multiGroup crossSectionSum_multiGroup
	./crossSection_multiGroup > o
	../Utilities/diff.com crossSection_multiGroup/crossSection_multiGroup crossSection_multiGroup.out o

	./crossSection_multiGroup --pid photon --tid O16 --map ../../../GIDI/Test/Data/MG_MC/all.map > o
	../Utilities/diff.com crossSection_multiGroup/crossSection_multiGroup-1 crossSection_multiGroup.photon+O16.out o
	./crossSection_multiGroup --pid photon --tid O16 --map ../../../GIDI/Test/Data/MG_MC/all.map -n > o
	../Utilities/diff.com crossSection_multiGroup/crossSection_multiGroup-2 crossSection_multiGroup.photon+O16.out o
	./crossSection_multiGroup --pid photon --tid O16 --map ../../../GIDI/Test/Data/MG_MC/all.map -a > o
	../Utilities/diff.com crossSection_multiGroup/crossSection_multiGroup-3 crossSection_multiGroup.photon+O16.atomic.out o
	./crossSection_multiGroup --pid photon --tid O16 --map ../../../GIDI/Test/Data/MG_MC/all.map -a -n > o
	../Utilities/diff.com crossSection_multiGroup/crossSection_multiGroup-4 crossSection_multiGroup.photon+O16.atomic+nuclear.out o

	./crossSectionSum_multiGroup > o
	../Utilities/diff.com crossSectionSum_multiGroup/crossSectionSum_multiGroup crossSectionSum_multiGroup.out o

clean:
	rm -f crossSection_multiGroup crossSectionSum_multiGroup o

realclean: clean
