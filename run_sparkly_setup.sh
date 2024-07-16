#!/bin/bash

SPARKLY_PATH_IN_DOCKER="/usr/src/sparkly"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <dataset> <config> [grid_config_folder] [results_folder] [base_path]" 
    echo "<dataset> - Key to extract dataset specific configuration"
    echo "<config> - Name of file containing relevant configurations"
    echo "[grid_config_folder] - Folder containing the config file (Optional)"
    echo "[results_folder] - Folder to store the configuration results (Optional)"
    echo "[base_path] - Path from which we reference the namespace (Optional)"
    exit 1
fi

DATASET=${1}
CONFIG=${2}
GRID_CONFIG_FOLDER=${3:-grid_config}
RESULTS_FOLDER=${4:-results}
BASE_PATH=${5:-$SCRIPT_DIR}

echo "Selected Setup:"
echo "<dataset> - $DATASET"
echo "<config> - $CONFIG"
echo "[grid_config_folder] - $GRID_CONFIG_FOLDER"
echo "[results_folder] - $RESULTS_FOLDER"
echo "[base_path] - $BASE_PATH"

chmod +x "${BASE_PATH}/run_sparkly_setup.py"
chmod +x "${BASE_PATH}/utilities/sparkly/gridsearch_utils.py"

DOCKER_RUN_CMD="docker run \
    -v ${BASE_PATH}/utilities/sparkly:${SPARKLY_PATH_IN_DOCKER}/utilities/sparkly \
    -v ${BASE_PATH}/gridsearch.sh:${SPARKLY_PATH_IN_DOCKER}/gridsearch.sh \
    -v ${BASE_PATH}/run_sparkly_setup.py:${SPARKLY_PATH_IN_DOCKER}/run_sparkly_setup.py \
    -v ${BASE_PATH}/utilities/sparkly/gridsearch_utils.py:${SPARKLY_PATH_IN_DOCKER}/gridsearch_utils.py \
    -v ${BASE_PATH}/datasets:${SPARKLY_PATH_IN_DOCKER}/data \
    -v ${BASE_PATH}/${GRID_CONFIG_FOLDER}:${SPARKLY_PATH_IN_DOCKER}/grid_config \
    -v ${BASE_PATH}/${RESULTS_FOLDER}:${SPARKLY_PATH_IN_DOCKER}/results \
    sparkly python3 run_sparkly_setup.py --dataset $DATASET --config $CONFIG"

eval "$DOCKER_RUN_CMD"


