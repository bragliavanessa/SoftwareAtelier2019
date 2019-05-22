#!/bin/bash

#python3 read_video.py
i=0
ii=0
while [ "$i" -lt 12 ]
do
    echo "$i"
    ii=$((i+1));
	python3 process_frame.py "$i" &
    python3 process_frame.py "$ii"
    
    wait
    i=$((ii+1));
done

python3 assemble_video.py