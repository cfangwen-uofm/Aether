#!/bin/sh

rm -rf ./run.test_*

# ----------------------------------------------------------------------
# run the acheron test:

cp -R ../../share/run ./run.test_acheron
cd run.test_acheron
cp ../aether.json.acheron ./aether.json
mpirun -np 6 ./aether

# post process and plot:
cd UA/output
python3 ../../../postAether.py -rm
# not sure where plotting code is located....
#/home/ridley/bin/run_plot_model_results.py 3DALL_20110320_*.nc -var="Temperature" -alt=2