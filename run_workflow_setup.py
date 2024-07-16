import time
import os
import sys
import pandas as pd
import numpy as np
import json
import argparse
from itertools import product
from pyjedai.utils import to_path
from pyjedai.datamodel import Data
from pyjedai.workflow import ProgressiveWorkFlow
from pyjedai.utils import (
    values_given,
    get_multiples,
    necessary_dfs_supplied,
    clear_json_file,
    purge_id_column,
    retrieve_top_workflows,
    workflows_to_dataframe)
from pyjedai.block_building import (
    StandardBlocking,
    QGramsBlocking,
    ExtendedQGramsBlocking,
    SuffixArraysBlocking,
    ExtendedSuffixArraysBlocking)
                                   
from pyjedai.block_cleaning import (
    BlockFiltering,
    BlockPurging)                         
from pyjedai.comparison_cleaning import (
    WeightedEdgePruning, 
    WeightedNodePruning,
    CardinalityEdgePruning,
    CardinalityNodePruning, 
    BLAST,
    ReciprocalCardinalityNodePruning,
    ReciprocalWeightedNodePruning,
    ComparisonPropagation)                                   
from pyjedai.prioritization import (
    GlobalTopPM, 
    LocalTopPM, 
    EmbeddingsNNBPM, 
    GlobalPSNM, 
    LocalPSNM, 
    PESM,
    class_references)
from pyjedai.evaluation import Evaluation



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    # Define arguments with default values
    parser.add_argument('--config_path',
                        dest='config_path',
                        type=str,
                        default="./grid_config/nn_experiments_top_1.json",
                        help='Path to the config file that stores the workflow setup to be executed')
    parser.add_argument('--store_folder_path',
                        dest='store_folder_path',
                        type=str,
                        default="./results/",
                        help='Path to the folder within which the workflow setup results will be stored')
    parser.add_argument('--dataset',
                        dest='dataset',
                        type=str,
                        default="d1",
                        help='Which dataset should the gridsearch be conducted for')
    parser.add_argument('--building',
                        dest='allow_building',
                        type=bool,
                        default=False,
                        help='Allow block building')
    parser.add_argument('--filtering',
                        dest='allow_filtering',
                        type=bool,
                        default=False,
                        help='Allow block filtering')
    parser.add_argument('--purging',
                        dest='allow_purging',
                        type=bool,
                        default=False,
                        help='Allow block purging')

    args = parser.parse_args()
    
    #-EDIT-THOSE-#
    # parameters native to the pyjedai progressive workflow
    # don't edit, unless new parameters were added to the workflow
    VALID_WORKFLOW_PARAMETERS = ['matcher',
                                'algorithm',
                                'number_of_nearest_neighbors',
                                'indexing',
                                'similarity_function',
                                'language_model',
                                'tokenizer',
                                'weighting_scheme',
                                'window_size',
                                'qgram']
    
    
    # path to the configuration file
    CONFIG_FILE_PATH = to_path(args.config_path)
    print(f"Config File Path: {CONFIG_FILE_PATH}")
    # path to the folder where the setup results will be stored
    STORE_FOLDER_PATH = to_path(args.store_folder_path)
    print(f"Store Folder Path: {STORE_FOLDER_PATH}")
    STORE_FILE_PATH = STORE_FOLDER_PATH + os.path.splitext(os.path.basename(CONFIG_FILE_PATH))[0]
    print(f"Store File Path: {STORE_FILE_PATH}")
    # path of the configuration file
    # CONFIG_FILE_PATH = to_path('~/best_nns/pyJedAI-Dev/script-configs/' + args.config_name + '.json')
    # which configuration from the json file should be used in current experiment  
    # EXPERIMENT_NAME = args.config_name + '_' + args.dataset
    # path at which the results will be stored within a json file
    # RESULTS_STORE_PATH = to_path('~/best_nns/pyJedAI-Dev/src/dbpedia_results/' + EXPERIMENT_NAME)
    JSON_STORE_PATH = STORE_FILE_PATH + '.json'
    DF_STORE_PATH = STORE_FILE_PATH + '.csv'
    # path at which the top workflows for specified argument values are stored
    BEST_WORKFLOWS_STORE_PATH = to_path(STORE_FILE_PATH + '_best.json')
    # results should be stored in the predefined path
    STORE_RESULTS = True
    STORE_RESULTS_AS_DF = True
    # AUC calculation and ROC visualization after execution
    VISUALIZE_RESULTS = True
    # workflow arguments and execution info should be printed in terminal once executed
    PRINT_WORKFLOWS = True
    # identifier column names for source and target datasets
    D1_ID = 'id'
    D2_ID = 'id'
    # specify which parameters affect the calculation of the distance matrix
    # construct the list of workflows in an order that allows for a distance matrix calculation, storage
    # and subsequent re-usage by workflows corresponding to all the combinations of its free (independent) parameters
    dm_params= ["similarity_function", "language_model", "tokenizer", "weighting_scheme", "qgram"]
    free_params = ["algorithm", "budget", "number_of_nearest_neighbors", "indexing"]
    # methods and their corresponding parameters for MB-based workflows
    # if you don't want to apply filtering, purging or block building (or want to use the default methods when necessary)
    # set those values to None
    _block_building = dict(method=StandardBlocking, 
                            params=dict()) if args.allow_building else None
    _block_filtering = dict(method=BlockFiltering, 
                            params=dict(ratio=0.8)) if args.allow_filtering else None

    _block_purging = dict(method=BlockPurging, 
                            params=dict(smoothing_factor=1.0)) if args.allow_purging else None  
    ##############                 
                                    
    with open(CONFIG_FILE_PATH) as file:
        config = json.load(file)
        
    config = config[args.dataset]

    if(not necessary_dfs_supplied(config)):
        raise ValueError("Different number of source, target dataset and ground truth paths!")

    datasets_info = list(zip(config['source_dataset_path'], config['target_dataset_path'], config['ground_truth_path']))
    iterations = config['iterations'][0] if(values_given(config, 'iterations')) else 1

    execution_count : int = 0

    if(STORE_RESULTS):
        clear_json_file(path=JSON_STORE_PATH)

    for id, dataset_info in enumerate(datasets_info):
        dataset_id = id + 1
        d1_path, d2_path, gt_path = dataset_info
        dataset_name = config['dataset_name'][id] if(values_given(config, 'dataset_name') and len(config['dataset_name']) > id) else ("D" + str(dataset_id))
        
        sep = config['separator'][id] if values_given(config, 'separator') else '|'
        d1 = pd.read_csv(to_path(d1_path), sep=sep, engine='python', na_filter=False).astype(str)
        d2 = pd.read_csv(to_path(d2_path), sep=sep, engine='python', na_filter=False).astype(str)
        gt = pd.read_csv(to_path(gt_path), sep=sep, engine='python')

        d1_attributes = config['d1_attributes'][id] if values_given(config, 'd1_attributes') else d1.columns.tolist()
        d2_attributes = config['d2_attributes'][id] if values_given(config, 'd2_attributes') else d2.columns.tolist()

        data = Data(
            dataset_1=d1,
            attributes_1=d1_attributes,
            id_column_name_1=D1_ID,
            dataset_2=d2,
            attributes_2=d2_attributes,
            id_column_name_2=D2_ID,
            ground_truth=gt,
        )
        
        true_positives_number = len(gt)
        workflow_config = {k: v for k, v in config.items() if(values_given(config, k) and k in VALID_WORKFLOW_PARAMETERS)}
        workflow_config['budget'] = config['budget'] if values_given(config, 'budget') else get_multiples(true_positives_number, 10)
        
        
        valid_dm_params = [pr for pr in dm_params if pr in workflow_config.keys()]
        valid_free_params = [pr for pr in free_params if pr in workflow_config.keys()]
        
        dm_config = {parameter: workflow_config[parameter] for parameter in valid_dm_params}
        free_config = {parameter: workflow_config[parameter] for parameter in valid_free_params}
        
        dm_arg_combinations = list(product(*list(dm_config.values())))
        free_arg_combinations = list(product(*list(free_config.values())))
        
        workflow_arg_combinations = []
        for dm_arg_combination in dm_arg_combinations:
            for free_arg_combination in free_arg_combinations:
                workflow_arg_combinations.append(dm_arg_combination + free_arg_combination)
        
        workflow_params = valid_dm_params + valid_free_params 
        total_workflows = len(workflow_arg_combinations) * len(datasets_info) * iterations
        
        for workflow_arg_combination in workflow_arg_combinations:
            workflow_arguments = dict(zip(workflow_params, workflow_arg_combination))
            workflow_arguments['dataset'] = dataset_name
            workflow_arguments['matcher'] = config['matcher'][id]
            print(workflow_arguments)
            
            for iteration in range(iterations):
                execution_count += 1
                print(f"#### WORKFLOW {execution_count}/{total_workflows} ####")
                current_workflow = ProgressiveWorkFlow()
                current_workflow.run(data=data,
                                    block_building=_block_building,
                                    block_purging=_block_purging,
                                    block_filtering=_block_filtering,
                                    **workflow_arguments)                        
                current_workflow_info = current_workflow.save(arguments=workflow_arguments,
                                                                path=JSON_STORE_PATH)
                if(PRINT_WORKFLOWS):
                    current_workflow.print_info(current_workflow_info)

    with open(JSON_STORE_PATH, 'r') as file:
        results = json.load(file)
                
    if(VISUALIZE_RESULTS):
        evaluator = Evaluation(data)
        evaluator.visualize_results_roc(results=results, drop_tp_indices=False)
        
    if(STORE_RESULTS):
        if(STORE_RESULTS_AS_DF):
            workflows_to_dataframe(workflows=results,
                                store_path=DF_STORE_PATH)
        else: 
            with open(JSON_STORE_PATH, 'w', encoding="utf-8") as file:
                json.dump(results, file, indent=4)
    
    
    
                                      
