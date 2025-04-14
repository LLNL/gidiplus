SHELL = /bin/sh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

DIRS = StyleGuide

.PHONY: default bin all docs check clean realclean doDIRS

default:

bin:

all:

docs:

check:

clean:
	$(MAKE) doDIRS _DIRS="$(DIRS)" _TARGET=clean

realclean:
	$(MAKE) doDIRS _DIRS="$(DIRS)" _TARGET=realclean

doDIRS:
	SAVED_PWD=`pwd`; \
	for DIR in $(_DIRS); do cd $$DIR; $(MAKE) $(_TARGET) CXXFLAGS="$(CXXFLAGS)"; cd $$SAVED_PWD; done
