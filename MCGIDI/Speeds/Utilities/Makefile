SHELL = /bin/ksh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

.PHONY: default check speeds clean realclean

local_CXXFLAGS = $(CXXFLAGS)

default: utilities4Speed.o

utilities4Speed.o: utilities4Speed.cpp utilities4Speed.hpp
	$(CXX) $(CXXFLAGS) -c utilities4Speed.cpp

check:

speeds: default

clean:
	rm -f *.o

realclean: clean
