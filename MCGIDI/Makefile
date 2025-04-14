SHELL = /bin/sh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

.PHONY: default all include lib bin speeds check clean realclean docs doDIRS

DIRS = Src include lib bin Test Speeds Doc

GIDI_PLUS_PATH ?= $(abspath ..)
export GIDI_PLUS_PATH

default: include lib
	cd Test; $(MAKE) default

all: default
	$(MAKE) bin
	$(MAKE) speeds

include:
	cd include; $(MAKE)

lib:
	cd lib; $(MAKE)

bin:
	cd bin; $(MAKE) bin

speeds:
	cd Speeds; $(MAKE) default

check:
	cd Test; $(MAKE) check

clean:
	$(MAKE) doDIRS _DIRS="$(DIRS)" _TARGET=clean

realclean:
	$(MAKE) doDIRS _DIRS="$(DIRS)" _TARGET=realclean

docs:
	doxygen Doc/doxygen/Doxyfile

doDIRS:
	SAVED_PWD=`pwd`; \
	for DIR in $(_DIRS); do cd $$DIR; $(MAKE) $(_TARGET); cd $$SAVED_PWD; done
