import time
import os
import sys
import pandas as pd
import numpy as np
import json
import argparse
import warnings
from itertools import product
from collections import defaultdict
from utilities.deepblocker.gridsearch_utils import (
    to_path,
    values_given,
    get_multiples,
    necessary_dfs_supplied,
    clear_json_file,
    purge_id_column,
    get_deepblocker_candidates,
    update_workflow_statistics,
    gt_to_df,
    iteration_normalized,
    get_valid_indexings,
    get_candidates_based_on_indexing,
    check_fasttext_dependencies
    )

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',
                        dest='config_path',
                        type=str,
                        default="./grid_config/deepblocker_experiments_top_1.json",
                        help='Path to the config file that stores the deepblocker setup to be executed')
    parser.add_argument('--store_folder_path',
                        dest='store_folder_path',
                        type=str,
                        default="./results/",
                        help='Path to the folder within which the deepblocker setup results will be stored')
    parser.add_argument('--dataset',
                        dest='dataset',
                        type=str,
                        default="d1",
                        help='Which dataset should the gridsearch be conducted for')
    args = parser.parse_args()
    
    #-EDIT-START-#
    # parameters native to the Deepblocker Workflow
    # don't edit, unless new parameters were added to the Workflow
    VALID_WORKFLOW_PARAMETERS = ["number_of_nearest_neighbors", "indexing"]
    
    
    # path to the configuration file
    CONFIG_FILE_PATH = to_path(args.config_path)
    print(f"Config File Path: {CONFIG_FILE_PATH}")
    # path to the folder where the setup results will be stored
    STORE_FOLDER_PATH = to_path(args.store_folder_path)
    print(f"Store Folder Path: {STORE_FOLDER_PATH}")
    STORE_FILE_PATH = STORE_FOLDER_PATH + os.path.splitext(os.path.basename(CONFIG_FILE_PATH))[0]
    print(f"Store File Path: {STORE_FILE_PATH}")
    # path at which the results will be stored within a json file
    RESULTS_STORE_PATH = to_path(STORE_FILE_PATH + '.csv')
    # path at which the top workflows for specified argument values are stored
    BEST_WORKFLOWS_STORE_PATH = to_path(STORE_FILE_PATH + '_best.json')
    # results should be stored in the predefined path
    STORE_RESULTS = True
    # AUC calculation and ROC visualization after execution
    VISUALIZE_RESULTS = True
    # workflow arguments and execution info should be printed in terminal once executed
    PRINT_WORKFLOWS = True
    # identifier column names for source and target datasets
    D1_ID = 'id'
    D2_ID = 'id'  
    #-EDIT-END-#          
    
    check_fasttext_dependencies()
    
    with open(CONFIG_FILE_PATH) as file:
        config = json.load(file)                           
    config = config[args.dataset]
    
    if(not necessary_dfs_supplied(config)):
        raise ValueError("Different number of source, target dataset and ground truth paths!")

    datasets_info = list(zip(config['source_dataset_path'], config['target_dataset_path'], config['ground_truth_path']))
    iterations = config['iterations'][0] if(values_given(config, 'iterations')) else 1
    execution_count : int = 0
    
    workflows_dataframe_columns = ['budget','dataset','total_candidates','total_emissions','time','auc','recall','tp_indices']
    workflows_dataframe_columns = VALID_WORKFLOW_PARAMETERS + workflows_dataframe_columns
    workflows_dataframe = pd.DataFrame(columns=workflows_dataframe_columns)

    for id, dataset_info in enumerate(datasets_info):
        dataset_id = id + 1
        
        
        dataset_indexing : str = config['indexing']
        
        
        d1_path, d2_path, gt_path = dataset_info
        dataset_name = config['dataset_name'][id] if(values_given(config, 'dataset_name') and len(config['dataset_name']) > id) else ("D" + str(dataset_id))
        sep = config['separator'][id] if values_given(config, 'separator') else '|'
        d1 : pd.DataFrame = pd.read_csv(to_path(d1_path), sep=sep, engine='python', na_filter=True).astype(str)
        d2 : pd.DataFrame = pd.read_csv(to_path(d2_path), sep=sep, engine='python', na_filter=True).astype(str)
        gt : pd.DataFrame = pd.read_csv(to_path(gt_path), sep=sep, engine='python')
        gt.columns = ['ltable_id', 'rtable_id']
        duplicate_of : dict = gt_to_df(ground_truth=gt)
        true_positives_number : int = len(gt)

        workflow_config : dict = {k: v for k, v in config.items() if(values_given(config, k) and k in VALID_WORKFLOW_PARAMETERS)}
        workflow_config['budget'] = config['budget'] if values_given(config, 'budget') else get_multiples(true_positives_number, 10)
        parameter_names : list = workflow_config.keys() 
        argument_combinations : list = list(product(*(workflow_config.values())))   
        total_workflows : int = len(argument_combinations) * len(datasets_info) * iterations
    
        for argument_combination in argument_combinations:
            workflow_arguments = dict(zip(parameter_names, argument_combination))
               
            workflow_statistics = defaultdict(float) 
            workflow_statistics['number_of_nearest_neighbors'] = workflow_arguments['number_of_nearest_neighbors']
            workflow_statistics['budget'] = workflow_arguments['budget']
            workflow_statistics['dataset'] = dataset_name
            workflow_statistics['indexing'] = workflow_arguments['indexing']
            valid_indexings : list = get_valid_indexings(indexing=workflow_statistics['indexing'])
                                
            for iteration in range(iterations):
                execution_count += 1
                print(f"#### WORKFLOW {execution_count}/{total_workflows} ####")
                start_time = time.time()
                
                inorder_candidates : pd.DataFrame = None
                reverse_candidates : pd.DataFrame = None
                for indexing in valid_indexings:
                    if(indexing == "inorder"):
                        inorder_candidates : pd.DataFrame = get_deepblocker_candidates(source_dataset=d1,
                                                                                       target_dataset=d2,
                                                                                       nearest_neighbors=workflow_arguments['number_of_nearest_neighbors'])
                    if(indexing == "reverse"):
                        reverse_candidates : pd.DataFrame = get_deepblocker_candidates(source_dataset=d2,
                                                                                       target_dataset=d1,
                                                                                       nearest_neighbors=workflow_arguments['number_of_nearest_neighbors'])
                        reverse_candidates['ltable_id'], reverse_candidates['rtable_id'] = reverse_candidates['rtable_id'], reverse_candidates['ltable_id']
                        
                
                candidates : pd.DataFrame = get_candidates_based_on_indexing(indexing=workflow_statistics['indexing'],
                                                                             inorder_candidates=inorder_candidates,
                                                                             reverse_candidates=reverse_candidates)
                
                update_workflow_statistics(statistics=workflow_statistics,
                                           candidates=candidates,
                                           ground_truth=gt,
                                           iterations=iterations,
                                           duplicate_of=duplicate_of)                        
                workflow_statistics['time'] += ((time.time() - start_time) / iterations)

            for column, value in workflow_statistics.items():
                workflow_statistics[column] = [value]            

            workflow_statistics = pd.DataFrame(workflow_statistics)
            workflow_statistics.reset_index(inplace=True, drop=True)
            workflows_dataframe = pd.concat([workflows_dataframe, pd.DataFrame(workflow_statistics)], axis=0, ignore_index=True)
               
            if(STORE_RESULTS):
                workflows_dataframe.to_csv(RESULTS_STORE_PATH, index=False)