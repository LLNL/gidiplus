SHELL = /bin/sh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

GIDI_PLUS_PATH ?= $(abspath ../..)

include $(GIDI_PLUS_PATH)/Makefile.paths

ifdef GIDI_PLUS_INCLUDE_PYTHON_BUILDING
	Python_CFLAGS = $(shell python3-config --cflags) -DGIDI_PLUS_INCLUDE_PYTHON_BUILDING
endif

.PHONY: default clean realclean

CppSources = $(sort $(wildcard *.cpp))
CppObjs = $(CppSources:.cpp=.o)
Archive = libGIDIP_python.a

default:
	echo ""
	echo "  -- Python/Src --"
	$(MAKE) $(Archive)

%.o: %.cpp
	echo "    "$<
	$(CXX) -c $(Python_CFLAGS) $(local_CXXFLAGS) -O1 $< -o $@

$(Archive): $(CppObjs)
	rm -f $@
	$(AR) -cr $@ $(CppObjs)

clean:
	rm -f *.o $(Archive)

realclean: clean
