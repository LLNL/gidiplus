SHELL = /bin/ksh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

.PHONY: default sampleProducts_multiGroup orphanGammas clean realclean

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

default: sampleProducts_multiGroup

sampleProducts_multiGroup: sampleProducts_multiGroup.cpp
	$(CXX) $(local_CXXFLAGS) -o sampleProducts_multiGroup sampleProducts_multiGroup.cpp $(LIBS)

orphanGammas: orphanGammas.cpp
	$(CXX) $(local_CXXFLAGS) -o orphanGammas orphanGammas.cpp $(LIBS)

check: sampleProducts_multiGroup
	./sampleProducts_multiGroup --all > o
	../Utilities/diff.com sampleProducts_multiGroup/sampleProducts_multiGroup Outs/sampleProducts.out o

	./sampleProducts_multiGroup --tid Th227 --all > o
	../Utilities/diff.com sampleProducts_multiGroup/sampleProducts_multiGroup Outs/sampleProducts.Th227.noDelayed.out o
	./sampleProducts_multiGroup --tid Th227 -d --all > o
	../Utilities/diff.com sampleProducts_multiGroup/sampleProducts_multiGroup Outs/sampleProducts.Th227.Delayed.out o
	./sampleProducts_multiGroup --tid Th227 -d > o
	../Utilities/diff.com sampleProducts_multiGroup/sampleProducts_multiGroup Outs/sampleProducts.Th227.n.Delayed.out o
	./sampleProducts_multiGroup --tid Th227 -d -p > o
	../Utilities/diff.com sampleProducts_multiGroup/sampleProducts_multiGroup Outs/sampleProducts.Th227.np.Delayed.out o
	./sampleProducts_multiGroup --tid Th227 -p > o
	../Utilities/diff.com sampleProducts_multiGroup/sampleProducts_multiGroup Outs/sampleProducts.Th227.np.noDelayed.out o

	./sampleProducts_multiGroup --pid photon --tid O16 --map ../../../GIDI/Test/Data/MG_MC/all.map --pa > o
	../Utilities/diff.com sampleProducts_multiGroup/sampleProducts_multiGroup Outs/sampleProducts.photon.pa.out o
	./sampleProducts_multiGroup --pid photon --tid O16 --map ../../../GIDI/Test/Data/MG_MC/all.map --pa -n > o
	../Utilities/diff.com sampleProducts_multiGroup/sampleProducts_multiGroup Outs/sampleProducts.photon.pa.out o
	./sampleProducts_multiGroup --pid photon --tid O16 --map ../../../GIDI/Test/Data/MG_MC/all.map --pn > o
	../Utilities/diff.com sampleProducts_multiGroup/sampleProducts_multiGroup Outs/sampleProducts.photon.pn.out o
	./sampleProducts_multiGroup --pid photon --tid O16 --map ../../../GIDI/Test/Data/MG_MC/all.map --pn -p > o
	../Utilities/diff.com sampleProducts_multiGroup/sampleProducts_multiGroup Outs/sampleProducts.photon.pn.photon.out o
	./sampleProducts_multiGroup --pid photon --tid O16 --map ../../../GIDI/Test/Data/MG_MC/all.map --pa --pn -p -n > o
	../Utilities/diff.com sampleProducts_multiGroup/sampleProducts_multiGroup Outs/sampleProducts.photon.pa.pn.neutron.photon.out o

	./sampleProducts_multiGroup --tid HinCH2 --map ../../../GIDI/Test/Data/MG_MC/neutrons/all.map > o
	../Utilities/diff.com sampleProducts_multiGroup/sampleProducts_multiGroup-HinCH2 Outs/sampleProducts.HinCH2.out o

clean:
	rm -f sampleProducts_multiGroup orphanGammas o

realclean: clean
