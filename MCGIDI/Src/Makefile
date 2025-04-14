SHELL = /bin/sh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

GIDI_PLUS_PATH ?= $(abspath ../..)

include $(GIDI_PLUS_PATH)/Makefile.paths

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
MCGIDI.hpp: MCGIDI_sampling.hpp MCGIDI_vector.hpp MCGIDI_string.hpp MCGIDI_functions.hpp MCGIDI_distributions.hpp MCGIDI_headerSource.hpp
MCGIDI_string.cpp: MCGIDI_string.hpp
MCGIDI_sampling.hpp: MCGIDI_vector.hpp

clean:
	rm -f $(CppObjs) $(Archive) MCGIDI.cpp

realclean: clean
