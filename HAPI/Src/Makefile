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

allCppSources = $(sort $(wildcard *.cpp))
allCppObjs = $(allCppSources:.cpp=.o)
ifdef HDF5_PATH
    CppSources = $(allCppSources)
else
    HDF_sources = $(wildcard *HDF*.cpp)
    CppSources = $(filter-out $(HDF_sources), $(allCppSources))
endif

CppObjs = $(CppSources:.cpp=.o)
Archive = libHAPI.a
Shared  = libHAPI.so

default:
	echo ""
	echo "  -- HAPI/Src --"
	$(MAKE) $(Archive)

%.o: %.cpp
	echo "    "$<
	$(CXX) -c $(local_CXXFLAGS) $< -o $@

$(Archive): $(CppObjs)
	rm -f $@
	$(AR) -cr $@ $(CppObjs)

$(Shared): $(CppObjs)
	rm -f $@
	$(CXX) -c -shared $(local_CXXFLAGS) $< -o $@

HAPI_Node.cpp : HAPI_Node_internal.cpp
HAPI_Data.cpp : HAPI_Data_internal.cpp
HAPI_PugiXMLNode.cpp : HAPI_Text.cpp HAPI_Data.cpp HAPI_Node.cpp HAPI.hpp
HAPI_HDFNode.cpp : HAPI_Text.cpp HAPI_Data.cpp HAPI_Node.cpp HAPI.hpp

clean:
	rm -f $(allCppObjs) $(Archive) $(Shared)

realclean: clean
