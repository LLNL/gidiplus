SHELL = /bin/ksh

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
Archive = libLUPI.a

default:
	echo ""
	echo "  -- LUPI/Src --"
	$(MAKE) $(Archive)

%.o: %.cpp
	echo "    "$<
	$(CXX) -c $(local_CXXFLAGS) $< -o $@

$(Archive): $(CppObjs)
	rm -f $@
	$(AR) -cr $@ $(CppObjs)

$(CppSources): LUPI.hpp

clean:
	rm -f *.o $(Archive)

realclean: clean
