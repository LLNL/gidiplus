SHELL = /bin/ksh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

include ../Makefile.paths

local_CXXFLAGS = $(CXXFLAGS) -I$(PUGIXML_INCLUDE) -I$(POPI_INCLUDE) -I$(SMR_INCLUDE) -I$(NF_INCLUDE)

.PHONY: default clean realclean

CppSources = $(sort $(wildcard *.cpp))
CppObjs = $(CppSources:.cpp=.o)
Archive = libGIDI.a

default:
	echo ""
	echo "  -- GIDI/Src --"
	$(MAKE) $(Archive)

%.o: %.cpp
	echo "    "$<
	$(CXX) -c $(local_CXXFLAGS) $< -o $@

$(Archive): $(CppObjs)
	rm -f $@
	$(AR) -cr $@ $(CppObjs)

GIDI.hpp: GIDI_data.hpp
$(CppSources): GIDI.hpp

clean:
	rm -f *.o $(Archive)

realclean: clean
