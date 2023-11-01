CURRENT_DIRECTORY=`pwd`

DATA_DIR=$CURRENT_DIRECTORY/data
OUTPUT_DIR=$CURRENT_DIRECTORY/output
MLFLOW_DIR=$CURRENT_DIRECTORY/mlruns
TB_DIR=$CURRENT_DIRECTORY/tensorboard

python ../smoke_main.py  --data $DATA_DIR \
                         --output $OUTPUT_DIR \
                         --mlflow-output=$MLFLOW_DIR \
                         --tensorboard-output=$TB_DIR \
                         --config config.yaml \
                         --disable-progressbar
