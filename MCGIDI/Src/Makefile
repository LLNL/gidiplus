SHELL = /bin/ksh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

.PHONY: default clean realclean

PIGI_PATH = ../../pugixml
PIGI_INCLUDE = $(PIGI_PATH)/include

POPs_PATH = ../../PoPI
POPs_INCLUDE = $(POPs_PATH)/include

SMR_PATH = ../../statusMessageReporting
SMR_INCLUDE = $(SMR_PATH)/include

NF_PATH = ../../numericalFunctions
NF_INCLUDE = $(NF_PATH)/include

GIDI_PATH = ../../GIDI
GIDI_INCLUDE = $(GIDI_PATH)/include

local_CXXFLAGS = $(CXXFLAGS) -I$(PIGI_INCLUDE) -I$(POPs_INCLUDE) -I$(SMR_INCLUDE) -I$(NF_INCLUDE) -I$(GIDI_INCLUDE)

CppSources = MCGIDI_settings.cpp \
             MCGIDI_protare.cpp \
             MCGIDI_protareComposite.cpp \
             MCGIDI_protareTNSL.cpp \
             MCGIDI_heatedCrossSections.cpp \
             MCGIDI_reaction.cpp \
             MCGIDI_outputChannel.cpp \
             MCGIDI_product.cpp \
             MCGIDI_misc.cpp \
             MCGIDI_domainHash.cpp \
             MCGIDI_functions.cpp \
             MCGIDI_sampling.cpp \
             MCGIDI_distributions.cpp \
             MCGIDI_string.cpp \
             MCGIDI_angle_biasing.cpp \
			 MCGIDI_nuclideGammaBranching.cpp \
			 MCGIDI_delayedNeutron.cpp \
			 MCGIDI_URR.cpp

CppObjs = $(CppSources:.cpp=.o)

default: $(CppObjs)
	rm -f libMCGIDI.a
	$(AR) -cr libMCGIDI.a *.o

%.o: %.cpp
	echo "    "$<
	$(CXX) -c $(local_CXXFLAGS) $< -o $@

clean:
	rm -f *.o *.so *.a

realclean: clean
