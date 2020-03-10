#!/bin/bash

# check if inkscape exists, exits if not
command -v inkscape
if [ "${?}" = 237 ]; then
	exit 0
fi

# loop through all *.svg files and remove their extension
for FIGURE in $(ls -1 figs/*.svg | sed -e 's/\..*$//' | sed -e 's#.*/##'); do

  old_md5=$(cat figs/$FIGURE.md5)
  current_md5=$(md5sum figs/$FIGURE.svg | cut -d' ' -f1)

  # check for changes in *.svg
  if [ "$old_md5" != "$current_md5" ]; then
    echo $current_md5 > figs/$FIGURE.md5
    inkscape --export-type="pdf" figs/$FIGURE.svg --export-latex
    sed -i.bak "s/$FIGURE/figs\/$FIGURE/" figs/$FIGURE.pdf_tex
    rm figs/*.bak
  fi

done
