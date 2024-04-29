TESTFOLDER=tests

all: main

main:
	latexmk -pdf $(LATEXMKSWITCHES) $@.tex

cleanmain: clean
	latexmk -pdf $(LATEXMKSWITCHES) -pretex="\def\main{1}" -usepretex main.tex

testcode: clean
	latexmk -pdf $(LATEXMKSWITCHES) -pretex="\def\testcode{1}" -usepretex main.tex

force:
	$(MAKE) LATEXMKSWITCHES=-gg all

clean:
	$(MAKE) LATEXMKSWITCHES=-C all

.PHONY: all main
