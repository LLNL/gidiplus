# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

SHELL = /bin/sh

SUB_DIRS = example1 example2

.PHONY: default check clean realclean

default:
	$(MAKE) MODE=default doSubDirs

check:
	$(MAKE) MODE=check doSubDirs

clean:
	$(MAKE) MODE=clean doSubDirs

realclean: clean

doSubDirs:
	SAVED_PWD=`pwd`; \
	for directory in $(SUB_DIRS); do cd $$directory; $(MAKE) $(MODE); cd $$SAVED_PWD; done
