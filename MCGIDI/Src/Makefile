SHELL = /bin/ksh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

include ../Makefile.paths

local_CXXFLAGS = $(CXXFLAGS) -I$(PUGIXML_INCLUDE) -I$(POPI_INCLUDE) -I$(SMR_INCLUDE) -I$(NF_INCLUDE) -I$(GIDI_INCLUDE)

.PHONY: default clean realclean

CppSources = $(sort $(wildcard *.cpp))
CppObjs = $(CppSources:.cpp=.o)
Archive = libMCGIDI.a

ifdef UNITY
CppObjs = MCGIDI.o
endif

default:
	echo ""
	echo "  -- MCGIDI/Src --"
	$(MAKE) $(Archive)

MCGIDI.cpp: $(CppSources)
	cat $(CppSources) > MCGIDI.cpp

%.o: %.cpp
	echo "    "$<
	$(CXX) -c $(local_CXXFLAGS) $< -o $@

$(Archive): $(CppObjs)
	rm -f $@
	$(AR) -cr $@ $(CppObjs)

$(CppSources): MCGIDI.hpp
MCGIDI.hpp: MCGIDI_sampling.hpp MCGIDI_dataBuffer.hpp MCGIDI_vector.hpp MCGIDI_string.hpp MCGIDI_declareMacro.hpp MCGIDI_functions.hpp MCGIDI_distributions.hpp
MCGIDI_string.cpp: MCGIDI_string.hpp
MCGIDI_dataBuffer.hpp MCGIDI_distributions.hpp MCGIDI_sampling.hpp MCGIDI_string.hpp MCGIDI_vector.hpp: MCGIDI_declareMacro.hpp
MCGIDI_sampling.hpp: MCGIDI_vector.hpp
MCGIDI_functions.hpp: MCGIDI_dataBuffer.hpp

clean:
	rm -f *.o $(Archive) MCGIDI.cpp

realclean: clean
