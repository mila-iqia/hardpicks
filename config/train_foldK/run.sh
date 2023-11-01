#!/bin/bash
#----------------------------------------------------------------------
# This script drives the execution of a FBP prediction run.
# It is meant to be executed on an EC2 instance that has been 
# properly provisioned with the code base and the data.
#----------------------------------------------------------------------
set -e

#shut_down_ec2_instance() {
#  echo "Shutting down EC2 instance in two minutes"
#  sudo shutdown -P +2
#}
#trap "shut_down_ec2_instance" ERR

EXPERIMENT_FAMILY="FBP"

export EXPERIMENT_NAME="train_foldK"

export USER='ec2-user' 

export REPO_DIR="/home/$USER/subsurface-rock-characterization"
export EXPERIMENT_DIR="/mnt/efs/$EXPERIMENT_FAMILY/experiments/$EXPERIMENT_NAME"

export DATA_BASE_DIR="$REPO_DIR/data/fbp/data"

export CONFIG=$EXPERIMENT_DIR/config.yaml

export LOG_DIR=$EXPERIMENT_DIR/logs
mkdir -p $LOG_DIR

timestamp=`date +%Y.%m.%d.%H.%M.%S`
host=`hostname`
log_filename=time_"$timestamp"_"$host".log

export OUTPUT=$EXPERIMENT_DIR/"output"
export MLFLOW=$EXPERIMENT_DIR/"mlruns"
export TENSORBOARD=$EXPERIMENT_DIR/"tensorboard"

# make sure we are in the right environment.
source /home/$USER/miniconda/etc/profile.d/conda.sh

conda activate hardpicks-dev

# switch to the repo directory so we can get the git hash.
cd $REPO_DIR

EXECUTABLE="$REPO_DIR/hardpicks/main.py"

PYTHON="$CONDA_PREFIX/bin/python"

$PYTHON $EXECUTABLE --data $DATA_BASE_DIR \
                    --output $OUTPUT\
                    --mlflow-output $MLFLOW \
                    --tensorboard-output $TENSORBOARD \
                    --config $CONFIG >& $LOG_DIR/$log_filename &

P1=$!

wait $P1

#shut_down_ec2_instance
