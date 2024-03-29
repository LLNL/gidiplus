# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

SHELL = /bin/sh

SMR_PATH = ../../../../statusMessageReporting
NUMERICALFUNCTIONS_PATH = ../../../..

ptwXYCFLAGS = -I$(NUMERICALFUNCTIONS_PATH)/include -L$(NUMERICALFUNCTIONS_PATH)/lib -lnumericalfunctions \
              -I$(SMR_PATH)/include -L$(SMR_PATH)/lib -lstatusMessageReporting \
              -lptwXY_utilities -lnfut_utilities
localCFLAGS = $(CFLAGS) $(ptwXYCFLAGS)

PYTHON = python3

TARGETS = flatToLinear flatToLinear2 flatToLinear3

.PHONY: default check clean realclean $(TARGETS)

default: $(TARGETS)

flatToLinear: flatToLinear.c
	$(CC) -o flatToLinear flatToLinear.c $(localCFLAGS) -lm

flatToLinear2: flatToLinear2.c
	$(CC) -o flatToLinear2 flatToLinear2.c $(localCFLAGS) -lm

flatToLinear3: flatToLinear3.c
	$(CC) -o flatToLinear3 flatToLinear3.c $(localCFLAGS) -lm

check: default
	for f in $(TARGETS); do ./$$f -v $(CHECKOPTIONS) > v; $(PYTHON) check.py $$f; done

clean:
	rm -f v

realclean: clean
	rm -fr $(TARGETS) *.dSYM
