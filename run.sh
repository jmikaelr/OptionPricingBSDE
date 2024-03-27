#!/bin/bash

wait_for_jobs() {
    local job
    for job in $(jobs -p); do
        wait $job || echo "Job $job exited with status $?"
    done
}

if [ -f output.log ]; then
    rm output.log
fi
readonly STOCK=100
readonly SIGMA=0.2
readonly MATURITY=0.25
readonly RATE=0.06
readonly RATE_1=0.01
readonly RATE_2=0.06
readonly STRIKE=100
readonly STRIKE_1=95
readonly STRIKE_2=105
readonly MEAN=0.05
readonly SAMPLES=50
readonly M_VALUES=(128 512 2048 8192 32768)

readonly N1=20
readonly N2=50
readonly N3=5
readonly N4=10

for M in "${M_VALUES[@]}"; do
     python3 main.py --S "$STOCK" --K "$STRIKE" --sigma "$SIGMA" --r "$RATE" --T "$MATURITY" --samples "$SAMPLES" --N "$N3" --M "$M" --plot --degrees 20 --opt_style european --nofig 
done

wait_for_jobs

for M in "${M_VALUES[@]}"; do
     python3 main.py --S "$STOCK" --K "$STRIKE" --sigma "$SIGMA" --r "$RATE" --T "$MATURITY" --samples "$SAMPLES" --N "$N4" --M "$M" --plot --degrees 20 --opt_style european --nofig 
done

wait_for_jobs

for M in "${M_VALUES[@]}"; do
     python3 main.py --S "$STOCK" --K "$STRIKE" --sigma "$SIGMA" --r "$RATE" --T "$MATURITY" --samples "$SAMPLES" --N "$N1" --M "$M" --plot --degrees 20 --opt_style american --nofig 
done

wait_for_jobs

for M in "${M_VALUES[@]}"; do
     python3 main.py --S "$STOCK" --K "$STRIKE" --sigma "$SIGMA" --r "$RATE" --T "$MATURITY" --samples "$SAMPLES" --N "$N2" --M "$M" --plot --degrees 20 --opt_style american --nofig 
done

wait_for_jobs

for M in "${M_VALUES[@]}"; do
     python3 main.py --S "$STOCK" --K "$STRIKE_1" --sigma "$SIGMA" --r "$RATE_1" --R "$RATE_2" --T "$MATURITY" --K2 "$STRIKE_2" --mu "$MEAN" --samples "$SAMPLES" --N "$N1" --M "$M" --plot --degrees 20 --opt_style europeanspread --nofig 
done

wait_for_jobs

for M in "${M_VALUES[@]}"; do
     python3 main.py --S "$STOCK" --K "$STRIKE_1" --sigma "$SIGMA" --r "$RATE_1" --R "$RATE_2" --T "$MATURITY" --K2 "$STRIKE_2" --mu "$MEAN" --samples "$SAMPLES" --N "$N2" --M "$M" --plot --degrees 20 --opt_style europeanspread --nofig 
done

wait_for_jobs

for M in "${M_VALUES[@]}"; do
     python3 main.py --S "$STOCK" --K "$STRIKE_1" --sigma "$SIGMA" --r "$RATE_1" --R "$RATE_2" --T "$MATURITY" --K2 "$STRIKE_2" --mu "$MEAN" --samples "$SAMPLES" --N "$N1" --M "$M" --plot --degrees 20 --opt_style americanspread --nofig 
done

wait_for_jobs

for M in "${M_VALUES[@]}"; do
     python3 main.py --S "$STOCK" --K "$STRIKE_1" --sigma "$SIGMA" --r "$RATE_1" --R "$RATE_2" --T "$MATURITY" --K2 "$STRIKE_2" --mu "$MEAN" --samples "$SAMPLES" --N "$N2" --M "$M" --plot --degrees 20 --opt_style americanspread --nofig 
done
