SHELL = /bin/sh

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

.PHONY: default all include lib bin check clean realclean doDIRS

DIRS = Src include lib

GIDI_PLUS_PATH ?= $(abspath ..)
export GIDI_PLUS_PATH

default: include lib

all: default

include:
	cd include; $(MAKE)

lib:
	cd lib; $(MAKE)

bin:

check:

clean:
	$(MAKE) doDIRS _DIRS="$(DIRS)" _TARGET=clean

realclean:
	$(MAKE) doDIRS _DIRS="$(DIRS)" _TARGET=realclean

doDIRS:
	SAVED_PWD=`pwd`; \
	for DIR in $(_DIRS); do cd $$DIR; $(MAKE) $(_TARGET); cd $$SAVED_PWD; done
