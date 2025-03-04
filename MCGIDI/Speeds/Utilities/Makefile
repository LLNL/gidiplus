SHELL = /bin/sh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

GIDI_PLUS_PATH ?= $(abspath ../../..)
CppSource2 = $(sort $(wildcard *.cpp))
CppObj2 = $(CppSource2:.cpp=.o)
Archive = libutilities4Speed.a

include $(GIDI_PLUS_PATH)/Makefile.paths
include ../Makefile.speeds

default: $(Archive)

$(Archive): $(CppObj2)
	rm -f $@
	$(AR) -cr $@ $(CppObj2)

speeds: $(Archive)

$(CppSource2): utilities4Speed.hpp

%.o: %.cpp
	echo "    "$@
	$(CXX) $(local_CXXFLAGS2) $< -c -o $@

check:
