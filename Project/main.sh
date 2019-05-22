#!/bin/bash

python3 read_video.py
i=0
ii=0
N=15
while [ "$i" -lt 3582 ]
do
    ii=$((i+N));
    if [ $ii -gt 3582 ] 
    then
        $ii = 3582
    fi
    for j in $(seq $i $ii)
    do
    python3 process_frame.py "$j" &
    done
    wait
    i=$((j+1));
done

python3 assemble_video.py