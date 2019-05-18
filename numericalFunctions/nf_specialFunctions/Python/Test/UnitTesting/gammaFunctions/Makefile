# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

PYTHON = python

.PHONY: default check clean realclean

default:

check:
	$(PYTHON) gammaTest.py
	$(PYTHON) incompleteGammaTest.py

checke:
	$(MAKE) check CHECKOPTIONS=-e

clean:

realclean: clean
