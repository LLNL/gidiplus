# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>

.PHONY: doc dvi pdf styleGuide.pdf clean realclean

doc: styleGuide.pdf

pdf: styleGuide.pdf
	pdflatex styleGuide.tex

styleGuide.pdf:
	pdflatex styleGuide.tex

clean:
	rm -f *.aux *.log *.lot *.toc

realclean: clean
	rm -f *.pdf
