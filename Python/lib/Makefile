# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

ARFLAGS = -cr

.PHONY: default libGIDIP_python.a check clean realclean

default: libGIDIP_python.a

libGIDIP_python.a:
	cd ../Src; $(MAKE)
	cp ../Src/libGIDIP_python.a .

check:

clean:
	rm -f *.o *.a *.so

realclean: clean
	rm -fr __.SYMDEF*
