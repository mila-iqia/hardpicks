#!/bin/bash

CURRENT_DIRECTORY=`pwd`
export ORION_DB_ADDRESS=$CURRENT_DIRECTORY/'orion_db.pkl'
export ORION_DB_TYPE='pickleddb'


DATA_DIR=$CURRENT_DIRECTORY/data
OUTPUT_DIR=$CURRENT_DIRECTORY/output
MLFLOW_DIR=$CURRENT_DIRECTORY/mlruns
TB_DIR=$CURRENT_DIRECTORY/tensorboard

PYTHON=/Users/bruno/miniconda3/envs/nrcan/bin/python


for gpu in `seq 0 3`; do
    echo "spawning worker $worker"
    orion -v hunt --config orion_config.yaml \
       $PYTHON ../smoke_main.py  --data $DATA_DIR \
                                 --output $OUTPUT_DIR/'{exp.working_dir}/{exp.name}_{trial.id}/' \
                                 --mlflow-output=$MLFLOW_DIR \
                                 --tensorboard-output=$TB_DIR \
                                 --config config.yaml \
                                 --gpu "gpu" \
                                 --disable-progressbar >& orion_worker_$gpu.log &
done
wait
