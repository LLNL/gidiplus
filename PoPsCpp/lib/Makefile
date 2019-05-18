# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

ARFLAGS = -cr

.PHONY: default libPoPs.a check clean realclean

default: libPoPs.a

libPoPs.a:
	cd ../Src; $(MAKE) CC=$(CXX) CXXFLAGS='$(CXXFLAGS)'
	cp ../Src/libPoPs.a .

check:

clean:
	rm -f *.o *.a *.so

realclean: clean
	rm -fr __.SYMDEF*
