# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

SHELL = /bin/sh

.PHONY: default utilities check checke checkPython clean realclean doTarget

subDirectories = Src lib include Test

default:
	cd lib; $(MAKE)
	cd include; $(MAKE)

utilities:
	cd Test; $(MAKE) utilities

check: default
	cd Test; $(MAKE) check

checke:
	$(MAKE) check CHECKOPTIONS=-e

checkPython:

clean:
	$(MAKE) doTarget MODE=clean

realclean:
	$(MAKE) doTarget MODE=realclean

doTarget:
	for directory in $(subDirectories); do cd $$directory; $(MAKE) $(MODE); cd ../; done
