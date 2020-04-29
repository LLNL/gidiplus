# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

SHELL = /bin/sh

subDirectories = GaussianQuadrature to_ptwXY normalize

.PHONY: default check clean realclean doTarget

default:
	$(MAKE) doTarget MODE=''

check: default
	-$(MAKE) doTarget MODE=check

clean:
	$(MAKE) doTarget MODE=clean

realclean: clean
	$(MAKE) doTarget MODE=realclean

doTarget:
	for directory in $(subDirectories); do cd $$directory; $(MAKE) $(MODE); cd ../; done
