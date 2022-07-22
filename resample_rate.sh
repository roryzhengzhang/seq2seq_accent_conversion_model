#!/bin/bash
name=NJS
for file in ~/Downloads/l2arctic_release_v5.0/${name}/wav/*.wav; do
    echo "$(basename $file)"
    mkdir $name
    sox -r 16k $file "$name/${name}_$(basename $file)"
done 