TESTFOLDER=tests

movefiles:
	mv main.* auxfiles/
	mv auxfiles/main.tex .
	mv auxfiles/main.pdf pdfs

main:
	latexmk -shell-escape -pdf -halt-on-error $(LATEXMKSWITCHES) $@.tex

full: clean main movefiles

testcode:
	latexmk --shell-escape pdf $(LATEXMKSWITCHES) -pretex="\def\testcode{1}" -usepretex main.tex

test: clean testcode movefiles

force:
	$(MAKE) LATEXMKSWITCHES=-gg all

clean:
	$(MAKE) LATEXMKSWITCHES=-C all

.PHONY: all main
