SHELL = /bin/sh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

GIDI_PLUS_PATH ?= $(abspath ../../..)
CppSource = $(sort $(wildcard *.cpp))

include $(GIDI_PLUS_PATH)/Makefile.paths

.PHONY: default check clean realclean

Executables = $(CppSource:.cpp=)

local_CXXFLAGS += -I$(GIDI_PATH)/Test/Utilities -I$(MCGIDI_PATH)/Test/Utilities
ifndef COMPILE_UTILITIES
    local_CXXFLAGS += $(LIB_PATHS) -L$(GIDI_PATH)/Test/Utilities -L$(MCGIDI_PATH)/Test/Utilities
endif

local_LDFLAGS_tmp = $(subst -x hip,,$(local_CXXFLAGS))
local_LDFLAGS = $(subst -x cu,,$(local_LDFLAGS_tmp))
# local_LDFLAGS := $(local_CXXFLAGS)
# LDFLAGS = local_CXXFLAGS .strip("-x hip")
# local_LDFLAGS =  $(LDFLAGS) $(LIB_PATHS) -L$(GIDI_PATH)/Test/Utilities -L$(MCGIDI_PATH)/Test/Utilities

LIBS =  -lGIDI_testUtilities -lMCGIDI_testUtilities $(MCGIDI_PATH)/Src/libMCGIDI.a -lGIDI -lPoPI -lHAPI -lLUPI -lGUPI -lRISI \
        -lnumericalfunctions -lstatusMessageReporting -lpugixml -lexpat -lm -lz -ldl $(HDF5_LIBS)

default: $(Executables)

$(Executables): %: %.o
	$(CXX) $(local_LDFLAGS) $< -o $@ $(LIBS)

clean:
	rm -rf $(Executables) $(Archive) *.o Outputs file1 file2

realclean: clean
	rm -rf *.dSYM

%.o: %.cpp
	$(CXX) -c $(local_CXXFLAGS) $< -o $@

check: gpuTest
	if [ ! -e Outputs ]; then mkdir Outputs; fi
	./gpuTest > Outputs/gpuTest.out
