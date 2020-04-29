# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

SHELL = /bin/sh

sourceDir = ../Src

.PHONY: default clean realclean

default: libnf_utilities.a

libnf_utilities.a: $(sourceDir)/*.c $(sourceDir)/*.h
	cd $(sourceDir); $(MAKE)
	$(AR) -cr libnf_utilities.a $(sourceDir)/*.o

clean:
	rm -f *.a

realclean: clean
