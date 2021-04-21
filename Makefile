SHELL = /bin/bash

# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

# These must be set by hand when we do a release.
gidiplus_major = 3
gidiplus_minor = 19

CXXFLAGS += -std=c++11

DIRS_GIDI_plus = numericalFunctions PoPI GIDI MCGIDI include lib Doc
DIRS = pugixml $(DIRS_GIDI_plus)

GIDI_PLUS_PATH ?= $(abspath .)
export GIDI_PLUS_PATH

PREFIX = `pwd`/install

.PHONY: default include lib pugixml pugixml_dummy install clean realclean tar doDIRS

default: pugixml
	@echo "Info GIDI_PLUS_PATH = $(GIDI_PLUS_PATH)"
	@echo "Info CXX            = $(CXX)"
	@echo "Info CXXFLAGS       = $(CXXFLAGS)"
	@echo "Info CC             = $(CC)"
	@echo "Info CFLAGS         = $(CFLAGS)"
	@echo "Info PREFIX         = $(PREFIX)"
	$(MAKE) doDIRS _DIRS="$(DIRS)" _TARGET=default

include:
	cd include; $(MAKE)

lib:
	cd lib; $(MAKE)

pugixml:
	rm -rf pugixml pugixml-1.8
	unzip -q Misc/pugixml-1.8.zip
	ln -s pugixml-1.8 pugixml
	cd pugixml; tar -xf ../Misc/pugixml.addon.tar

pugixml_dummy:
	rm -rf pugixml-1.8 pugixml
	mkdir pugixml
	cp Misc/Makefile_dummy pugixml/Makefile

gidiplus_version.h: FORCE
	@if test -d "./.git"; then \
		echo "#define GIDIPLUS_MAJOR ${gidiplus_major}"  > gidiplus_version.h; \
		echo "#define GIDIPLUS_MINOR ${gidiplus_minor}" >> gidiplus_version.h; \
		git describe | awk -F "-" '{if (NF>2) {printf("#define GIDIPLUS_PATCHLEVEL %d\n", $$(NF-1))} else {print "#define GIDIPLUS_PATCHLEVEL 0"}}' >> gidiplus_version.h; \
		git rev-parse HEAD                     | awk '{printf("#define GIDIPLUS_GIT %s\n" , $$1)}'                 >> gidiplus_version.h; \
		git ls-files -s numericalFunctions     | awk '{printf("#define NUMERICAL_FUNCTIONS_GIT %s\n" , $$2)}'      >> gidiplus_version.h; \
		git ls-files -s PoPI                   | awk '{printf("#define POPI_GIT %s\n" , $$2)}'                     >> gidiplus_version.h; \
		git ls-files -s GIDI                   | awk '{printf("#define GIDI_GIT %s\n" , $$2)}'                     >> gidiplus_version.h; \
		git ls-files -s MCGIDI                 | awk '{printf("#define MCGIDI_GIT %s\n" , $$2)}'                   >> gidiplus_version.h; \
	fi
	cat gidiplus_version.h

install: default
	mkdir -p $(PREFIX)/include
	cp -p include/* $(PREFIX)/include
	mkdir -p $(PREFIX)/lib
	cp -p lib/libgidiplus.a $(PREFIX)/lib

docs:
	$(MAKE) doDIRS _DIRS="$(DIRS_GIDI_plus)" _TARGET=docs

check:
	$(MAKE) doDIRS _DIRS="$(DIRS)" _TARGET=check

clean: FORCE 
	if [ ! -e pugixml ]; then $(MAKE) pugixml_dummy; fi
	$(MAKE) doDIRS _DIRS="$(DIRS)" _TARGET=clean

realclean: pugixml_dummy
	$(MAKE) doDIRS _DIRS="$(DIRS)" _TARGET=realclean
	rm -rf pugixml test_install

tar: gidiplus_version.h
	$(MAKE) -s realclean
	fileName=`git describe | awk -F "-" '{(NF>2) ? patchVal=$$(NF-1) : patchVal=0; printf "gidiplus-%s.%s.%s.%s", ${gidiplus_major}, ${gidiplus_minor}, patchVal, $$NF}'`; \
	rm -rf ../$$fileName; \
	mkdir ../$$fileName; \
	absolutePath=`cd ../$$fileName; pwd`; \
	git archive --format=tar HEAD | (cd ../$$fileName && tar -xf -); \
	git submodule foreach --recursive "git archive --prefix=\$$displaypath/ --format=tar HEAD | (cd $$absolutePath && tar -xf -)"; \
	cp -r GIDI/Test/Data ../$$fileName/GIDI/Test; \
	find ../$$fileName -iname ".git*" -exec rm {} \; ; \
	find ../$$fileName -iname "Makefile.popskit" -exec rm {} \; ; \
	rm -rf ../$${fileName}/custom_hooks ../$${fileName}/GIDI/Test/GIDI_Test_Data.tar.gz; \
	cp gidiplus_version.h ../$$fileName; \
	cd ../; \
	tar -cf $${fileName}.tar $$fileName

doDIRS:
	SAVED_PWD=`pwd`; \
	for DIR in $(_DIRS); do cd $$DIR; $(MAKE) $(_TARGET) CXXFLAGS="$(CXXFLAGS)"; cd $$SAVED_PWD; done

FORCE:
