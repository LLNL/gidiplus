SHELL = /bin/ksh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

.PHONY: default settingsProcess productMatrixCollapse productMatrixCollapse_photoAtomic clean realclean

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
		-I../Utilities -L../Utilities \
		-I../../include -L../../lib \
		-I$(POPs_INCLUDE) -L$(POPs_LIB) \
		-I$(SMR_INCLUDE) -L$(SMR_LIB) \
		-I$(NF_INCLUDE) -L$(NF_LIB) \
		-I$(PUGIXML_INCLUDE) -L$(PUGIXML_LIB)

LIBS = -lGIDI_testUtilities -lGIDI -lPoPI -lstatusMessageReporting -lnumericalfunctions -lpugixml

default: settingsProcess productMatrixCollapse productMatrixCollapse_photoAtomic

settingsProcess: settingsProcess.cpp
	$(CXX) $(local_CXXFLAGS) -o settingsProcess settingsProcess.cpp $(LIBS)

productMatrixCollapse: productMatrixCollapse.cpp
	$(CXX) $(local_CXXFLAGS) -o productMatrixCollapse productMatrixCollapse.cpp $(LIBS)

productMatrixCollapse_photoAtomic: productMatrixCollapse_photoAtomic.cpp
	$(CXX) $(local_CXXFLAGS) -o productMatrixCollapse_photoAtomic productMatrixCollapse_photoAtomic.cpp $(LIBS)

check: settingsProcess productMatrixCollapse productMatrixCollapse_photoAtomic
	./settingsProcess --tid O16 > o1
	python ../Utilities/diff.py settingsProcess/settingsProcess O16.out o1
	./settingsProcess --tid Th227 > o2
	python ../Utilities/diff.py settingsProcess/settingsProcess Th227.out o2
	./productMatrixCollapse --tid O16 > o3
	python ../Utilities/diff.py settingsProcess/productMatrixCollapse productMatrixCollapse.O16.out o3
	./productMatrixCollapse --tid Th227 > o4
	python ../Utilities/diff.py settingsProcess/productMatrixCollapse productMatrixCollapse.Th227.out o4
	./productMatrixCollapse_photoAtomic > o5
	python ../Utilities/diff.py settingsProcess/productMatrixCollapse_photoAtomic productMatrixCollapse_photoAtomic.O.out o5

clean:
	rm -f settingsProcess productMatrixCollapse productMatrixCollapse_photoAtomic o[0-9]* file1 file2

realclean: clean
	rm -rf *.dSYM
