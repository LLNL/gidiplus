# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

ARFLAGS = -cr

.PHONY: default libRISI.a check clean realclean

default: libRISI.a

libRISI.a:
	cd ../Src; $(MAKE)
	cp ../Src/libRISI.a .

check:

clean:
	rm -f *.o *.a *.so

realclean: clean
	rm -fr __.SYMDEF*
