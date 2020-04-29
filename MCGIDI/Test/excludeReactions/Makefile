SHELL = /bin/ksh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

.PHONY: default excludeReactions clean realclean

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

default: excludeReactions

excludeReactions: excludeReactions.cpp
	$(CXX) $(local_CXXFLAGS) -o excludeReactions excludeReactions.cpp $(LIBS)

check: excludeReactions
	./excludeReactions > o
	../Utilities/diff.com excludeReactions/excludeReactions excludeReactions.out o
	./excludeReactions 2 > o
	../Utilities/diff.com excludeReactions/excludeReactions excludeReactions_2.out o
	./excludeReactions 5 > o
	../Utilities/diff.com excludeReactions/excludeReactions excludeReactions_5.out o
	./excludeReactions 0 2 5 > o
	../Utilities/diff.com excludeReactions/excludeReactions excludeReactions_0_2_5.out o

	./excludeReactions --pid n --tid O16 > o
	../Utilities/diff.com excludeReactions/excludeReactions excludeReactions_n_O16.out o
	./excludeReactions --pid n --tid O16 3 4 5 6 7 8 > o
	../Utilities/diff.com excludeReactions/excludeReactions excludeReactions_n_O16_3-8.out o
	./excludeReactions --pid n --tid O16 3 4 5 6 7 8 17 18 19 20 21 22 23 24 > o
	../Utilities/diff.com excludeReactions/excludeReactions excludeReactions_n_O16_3-8_17-24.out o

	./excludeReactions --pid n --tid tnsl-Al27 --map ../../../GIDI/Test/Data/MG_MC/neutrons/all.map > o

clean:
	rm -f excludeReactions o

realclean: clean
