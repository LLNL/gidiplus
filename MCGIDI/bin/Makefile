SHELL = /bin/sh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

.PHONY: default bin compile check clean realclean

default: bin

bin:
	cd Src; $(MAKE) bin

compile: default

check:

clean:
	cd Src; $(MAKE) clean

realclean:
	cd Src; $(MAKE) realclean
