#!/bin/bash

# check if inkscape exists, exits if not
command -v inkscape
if [ "${?}" = 237 ]; then
	exit 0
fi

# loop through all *.svg files and remove their extension
for FIGURE in $(ls -1 figs/*.svg | sed -e 's/\..*$//' | sed -e 's#.*/##'); do
	inkscape --export-type="pdf" figs/$FIGURE.svg --export-latex
	sed -i.bak "s/$FIGURE/figs\/$FIGURE/" figs/$FIGURE.pdf_tex
	rm figs/*.bak
done
