SHELL = /bin/ksh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

.PHONY: default productMatrix clean realclean

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

local_CXXFLAGS = $(CXXFLAGS) -I../../include -L../../lib \
		-I../Utilities -L../Utilities \
		-I$(POPs_INCLUDE) -L$(POPs_LIB) \
		-I$(SMR_INCLUDE) -L$(SMR_LIB) \
		-I$(NF_INCLUDE) -L$(NF_LIB) \
        -I$(PUGIXML_INCLUDE) -L$(PUGIXML_LIB)

LIBS = -lGIDI_testUtilities -lGIDI -lPoPI -lstatusMessageReporting -lnumericalfunctions -lpugixml

default: productMatrix

productMatrix: productMatrix.cpp
	$(CXX) $(local_CXXFLAGS) -o productMatrix productMatrix.cpp $(LIBS)

check: productMatrix
	./productMatrix --tid O16 > o1
	python ../Utilities/diff.py "productMatrix/productMatrix O16" O16.out o1
	./productMatrix --tid Th227 > o2
	python ../Utilities/diff.py "productMatrix/productMatrix Th227" Th227.noDelayed.out o2
	./productMatrix --tid Th227 --delayed > o3
	python ../Utilities/diff.py "productMatrix/productMatrix Th227" Th227.delayed.out o3

clean:
	rm -f productMatrix o[0-9]* file1 file2

realclean: clean
	rm -rf *.dSYM
