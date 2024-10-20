LTS_overview:
	latexmk -shell-escape -pdf -halt-on-error $(LATEXMKSWITCHES) $@.tex

full: clean LTS_overview 

force:
	$(MAKE) LATEXMKSWITCHES=-gg all

clean:
	$(MAKE) LATEXMKSWITCHES=-C all

.PHONY: all main
