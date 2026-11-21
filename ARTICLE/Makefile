# ARTICLE/Makefile — build the article and gate the result
TEX      := article
LATEXMK  := latexmk
PDF      := $(TEX).pdf
SOURCES  := $(TEX).tex $(wildcard sections/*.tex) refs.bib

.PHONY: all clean check view

all: $(PDF)

$(PDF): $(SOURCES)
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error $(TEX).tex

# Fail on unresolved references/citations. latexmk returns 0 with undefined
# references, so read the log instead of trusting the exit code.
check: $(PDF)
	@if grep -Eq "undefined references|undefined citations|Citation .* undefined|Reference .* undefined" $(TEX).log; then \
	  echo "FAIL: undefined references or citations"; \
	  grep -E "undefined references|undefined citations|Citation .* undefined|Reference .* undefined" $(TEX).log; \
	  exit 1; \
	fi
	@if grep -Eq "Overfull \\\\hbox \([1-9][0-9]+\.[0-9]+pt" $(TEX).log; then \
	  echo "WARN: overfull boxes over 10pt"; \
	  grep -E "Overfull \\\\hbox" $(TEX).log | head; \
	fi
	@echo "OK: $(PDF) built with no undefined references or citations"

view: $(PDF)
	@command -v xdg-open >/dev/null && xdg-open $(PDF) || echo "open $(PDF)"

clean:
	$(LATEXMK) -C $(TEX).tex
	rm -f $(TEX).bbl $(TEX).run.xml
