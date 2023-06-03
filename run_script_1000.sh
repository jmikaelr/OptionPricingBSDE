#!/bin/bash

echo "Runtime,OptionPrice,LowerBoundOpt,UpperBoundOpt,HedgeRatio,LowerBoundHedge,UpperBoundHedge" > output.csv

for i in {1..1000}
do
    python2 main.py >> output.csv
done

