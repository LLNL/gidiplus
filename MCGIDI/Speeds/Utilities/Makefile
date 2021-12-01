SHELL = /bin/ksh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

GIDI_PLUS_PATH ?= $(abspath ../../..)
CppSources = $(sort $(wildcard *.cpp))
CppObjs = $(CppSources:.cpp=.o)
Archive = libutilities4Speed.a

include $(GIDI_PLUS_PATH)/Makefile.paths
include ../Makefile.speeds

default: $(Archive)

$(Archive): $(CppObjs)
	rm -f $@
	$(AR) -cr $@ $(CppObjs)

speeds: $(Archive)

$(CppSources): utilities4Speed.hpp

check:
