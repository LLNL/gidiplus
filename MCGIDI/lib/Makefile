# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

ARFLAGS = -cr

.PHONY: default libMCGIDI.a check clean realclean

default: libMCGIDI.a

libMCGIDI.a:
	cd ../Src; $(MAKE) CC=$(CXX) CXXFLAGS='$(CXXFLAGS)' POPs_PATH='$(POPs_PATH)' SMR_PATH='$(SMR_PATH)' NF_PATH='$(NF_PATH)'
	cp ../Src/libMCGIDI.a .

check:

clean:
	rm -f *.o *.a *.so

realclean: clean
	rm -fr __.SYMDEF*
