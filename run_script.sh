#!/bin/bash

echo "Runtime,OptionPrice,LowerBoundOpt,UpperBoundOpt,HedgeRatio,LowerBoundHedge,UpperBoundHedge" > output.csv

runs=$1

for i in $(seq 1 $runs)
do
    python2 main.py >> output.csv
done

