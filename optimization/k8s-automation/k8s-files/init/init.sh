#!/bin/sh

files=$(ls -p /etc/smart-tuning-ro | grep -v /)

echo "Copying env variables to read-write space"
for var in $files; do
  # avoid trying to copy dot dirs
  if [[ ! -d $var ]]; then
    echo -e "  >>> $var=$(cat /etc/smart-tuning-ro/$var)"
    cp /etc/smart-tuning-ro/$var /etc/smart-tuning-rw/
  fi
done

echo -e "Creating smart-tuning.env"
for var in $files; do
  # avoid trying to copy dot dirs
  if [[ ! -d $var ]]; then
    echo -e "  >>> $var=$(cat /etc/smart-tuning-rw/$var)"
    echo -e $var=$(cat /etc/smart-tuning-rw/$var) >> /etc/smart-tuning-rw/smart-tuning.env
  fi
done

exit 0