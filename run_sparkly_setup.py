import os
import pandas as pd
from time import time
import json
import sys
sys.path.append('.')
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from sparkly.index import IndexConfig, LuceneIndex
from sparkly.search import Searcher
from pathlib import Path


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
from utilities.sparkly.gridsearch_utils import (
    to_path,
    values_given,
    get_multiples,
    necessary_dfs_supplied,
    clear_json_file,
    purge_id_column,
    # get_deepblocker_candidates,
    update_workflow_statistics,
    gt_to_df,
    iteration_normalized,
    get_valid_indexings,
    get_candidates_based_on_indexing
    )

def run_sparkly(index, query, gt, sep, cid, tid, limit, dataset):
    # path to the test data
    # data_path = Path('./examples/data/abt_buy/').absolute()
    # # table to be indexed
    # table_a_path = data_path / 'table_a.csv'
    # # table for searching
    # table_b_path = data_path / 'table_b.csv'
    # # the ground truth
    # gold_path = data_path / 'gold.csv'
    # the analyzers used to convert the text into tokens for indexing
    analyzers = ['3gram']
    
    # initialize a local spark context
    spark = SparkSession.builder\
                        .master('local[*]')\
                        .appName('Sparkly Example')\
                        .getOrCreate()
    # read all the data as spark dataframes with tab as the separator
    sep = sep.replace("\\", "") 
    table_a = spark.read.csv(index, header=True, inferSchema=True, sep=sep)
    table_a = table_a.withColumnRenamed(cid, "_id")
    table_b = spark.read.csv(query, header=True, inferSchema=True, sep=sep)
    table_b = table_b.withColumnRenamed(cid, "_id")
    cid = "_id"
    gold = spark.read.csv(gt, header=True, inferSchema=True, sep=sep)
    # the index config, '_id' column will be used as the unique 
    # id column in the index. Note id_col must be an integer (32 or 64 bit)
    config = IndexConfig(id_col=cid)
    # add the 'name' column to be indexed with analyzer above
    config.add_field(tid, analyzers)
    # create a new index stored at /tmp/example_index/
    index = LuceneIndex(f'/tmp/example_index_{dataset}/', config)
    # index the records from table A according to the config we created above
    index.upsert_docs(table_a)
    
    # get a query spec (template) which searches on 
    # all indexed fields
    query_spec = index.get_full_query_spec()
    # create a searcher for doing bulk search using our index
    searcher = Searcher(index)
    # search the index with table b
    candidates = searcher.search(table_b, query_spec, id_col=cid, limit=limit).cache()
    ids_exploded = candidates.selectExpr("_id", "posexplode_outer(ids) AS (pos, id2)")
    scores_exploded = candidates.selectExpr("_id", "posexplode_outer(scores) AS (pos, similarity)")
    # join ID and score exploded DataFrames based on the position index
    candidates_df = ids_exploded.join(scores_exploded, ["_id", "pos"]).drop("pos")
    
    candidates_df = candidates_df.withColumnRenamed("_id", "rtable_id").withColumnRenamed("id2", "ltable_id").toPandas()    
    candidates.unpersist()   
    return candidates_df
    
data_directory = '/usr/src/sparkly/data/'
# log_file = '/usr/src/sparkly/logs/Sparkly.txt'

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name',
                        dest='config_name',
                        type=str,
                        default="deepblocker_experiments_top_1",
                        help='Name of the configuration from grid_config folder that we want to execute')
    parser.add_argument('--dataset',
                        dest='dataset',
                        type=str,
                        default="d1",
                        help='Which dataset should the gridsearch be conducted for')
    args = parser.parse_args()
    
    
    #-EDIT-START-#
    # parameters native to the Deepblocker Workflow
    # don't edit, unless new parameters were added to the Workflow
    EXECUTION_PATH = '/usr/src/sparkly/'
    VALID_WORKFLOW_PARAMETERS = ["number_of_nearest_neighbors", "indexing"]
    # path of the configuration file
    CONFIG_FILE_PATH = to_path(EXECUTION_PATH + 'grid_config/' + args.config_name + '.json')
    # which configuration from the json file should be used in current experiment  
    EXPERIMENT_NAME = args.config_name + '_' + args.dataset
    # path at which the results will be stored within a json file
    RESULTS_STORE_PATH = to_path(EXECUTION_PATH + 'results/' + EXPERIMENT_NAME + '.csv')
    # path at which the top workflows for specified argument values are stored
    BEST_WORKFLOWS_STORE_PATH = to_path(RESULTS_STORE_PATH + '_best.json')
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
        d1_path, d2_path, gt_path = dataset_info
        dataset_name = config['dataset_name'][id] if(values_given(config, 'dataset_name') and len(config['dataset_name']) > id) else ("D" + str(dataset_id))
        
        sep = config['separator'][id] if values_given(config, 'separator') else '|'
        d1 : pd.DataFrame = pd.read_csv(to_path(d1_path), sep=sep, engine='python', na_filter=False).astype(str)
        d2 : pd.DataFrame = pd.read_csv(to_path(d2_path), sep=sep, engine='python', na_filter=False).astype(str)
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
                
                ground = 'file://{}'.format(gt_path)
                query = 'file://{}'.format(d2_path)
                index = 'file://{}'.format(d1_path)
                cid, tid = 'id', 'aggregate value'
                inorder_candidates : pd.DataFrame = None
                reverse_candidates : pd.DataFrame = None
                
                for indexing in valid_indexings:
                    if(indexing == "inorder"):
                        inorder_candidates : pd.DataFrame = run_sparkly(index=index, query=query,
                                                            gt=ground, sep=sep,
                                                            cid=cid, tid=tid, 
                                                            limit=workflow_arguments['number_of_nearest_neighbors'],
                                                            dataset=args.dataset)
                    if(indexing == "reverse"):
                        reverse_candidates : pd.DataFrame = run_sparkly(index=query, query=index,
                                                            gt=ground, sep=sep,
                                                            cid=cid, tid=tid, 
                                                            limit=workflow_arguments['number_of_nearest_neighbors'],
                                                            dataset=args.dataset)
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
