SHELL = /bin/ksh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

.PHONY: default sampleProducts orphanGammas clean realclean

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

default: sampleProducts

sampleProducts: sampleProducts.cpp
	$(CXX) $(local_CXXFLAGS) -o sampleProducts sampleProducts.cpp $(LIBS)

orphanGammas: orphanGammas.cpp
	$(CXX) $(local_CXXFLAGS) -o orphanGammas orphanGammas.cpp $(LIBS)

check: sampleProducts
	./sampleProducts --all > o01
	../Utilities/diff.com sampleProducts/sampleProducts sampleProducts.out o01

	./sampleProducts --tid Th227 --all > o02
	../Utilities/diff.com sampleProducts/sampleProducts sampleProducts.Th227.noDelayed.out o02
	./sampleProducts --tid Th227 -d --all > o03
	../Utilities/diff.com sampleProducts/sampleProducts sampleProducts.Th227.Delayed.out o03
	./sampleProducts --tid Th227 -d > o04
	../Utilities/diff.com sampleProducts/sampleProducts sampleProducts.Th227.n.Delayed.out o04
	./sampleProducts --tid Th227 -d -p > o05
	../Utilities/diff.com sampleProducts/sampleProducts sampleProducts.Th227.np.Delayed.out o05
	./sampleProducts --tid Th227 -p > o06
	../Utilities/diff.com sampleProducts/sampleProducts sampleProducts.Th227.np.noDelayed.out o06

	./sampleProducts --pid photon --tid O16 --map ../../../GIDI/Test/Data/MG_MC/all.map --pa > o07
	../Utilities/diff.com sampleProducts/sampleProducts sampleProducts.photon.pa.out o07
	./sampleProducts --pid photon --tid O16 --map ../../../GIDI/Test/Data/MG_MC/all.map --pa -n > o08
	../Utilities/diff.com sampleProducts/sampleProducts sampleProducts.photon.pa.out o08
	./sampleProducts --pid photon --tid O16 --map ../../../GIDI/Test/Data/MG_MC/all.map --pn > o09
	../Utilities/diff.com sampleProducts/sampleProducts sampleProducts.photon.pn.out o09
	./sampleProducts --pid photon --tid O16 --map ../../../GIDI/Test/Data/MG_MC/all.map --pn -p > o10
	../Utilities/diff.com sampleProducts/sampleProducts sampleProducts.photon.pn.photon.out o10
	./sampleProducts --pid photon --tid O16 --map ../../../GIDI/Test/Data/MG_MC/all.map --pa --pn -p -n > o11
	../Utilities/diff.com sampleProducts/sampleProducts sampleProducts.photon.pa.pn.neutron.photon.out o11

	./sampleProducts --tid HinCH2 --map ../../../GIDI/Test/Data/MG_MC/neutrons/all.map > o12
	../Utilities/diff.com sampleProducts/sampleProducts-HinCH2 sampleProducts.HinCH2.out o12

clean:
	rm -f sampleProducts orphanGammas o[0-9]*

realclean: clean
