# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

SHELL = /bin/sh

PYTHON = python3

.PHONY: default check clean realclean

default:

check:
	$(PYTHON) getSlice.py
	$(PYTHON) setSlice.py
	$(PYTHON) xSlice.py

checke:
	$(MAKE) CHECKOPTIONS="-v" check

clean:

realclean: clean
