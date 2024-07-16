import os
import sys
import json
import pandas as pd
from collections import defaultdict
import time
from .deep_blocker import DeepBlocker
from .tuple_embedding_models import  AutoEncoderTupleEmbedding
from .vector_pairing_models import ExactTopKVectorPairing
from . import blocking_utils
from . import configurations
import zipfile
import subprocess
from pathlib import Path
from . import utils
from .utils import cases
from .configurations import *

def unzip_file(zip_path : str, extract_to : str) -> None:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        
def check_fasttext_dependencies() -> None:
    fasttext_dir = Path.home() / "fasttext"
    zip_file_path = fasttext_dir / "wiki.en.zip"
    bin_file_path = fasttext_dir / "wiki.en.bin"
    download_url = FASTTEXT_EMBEDDIG_DOWNLOAD_URL
    
    fasttext_dir.mkdir(parents=True, exist_ok=True)
    
    if bin_file_path.exists():
        print("wiki.en.bin found. Granting Permissions To Bin File...")
        subprocess.run(['chmod', '-R', '+x', str(bin_file_path)])
        print("Permissions granted. Continuing...")
    elif zip_file_path.exists():
        print("wiki.en.zip found. Unzipping...")
        unzip_file(zip_file_path, fasttext_dir)
        print("Unzipping completed.")
    else:
        while True:
            user_input = input(f"No wiki.en.zip or wiki.en.bin found. Do you want to download the file? (yes/no): ").strip().lower()
            if user_input == 'yes':
                print("Downloading wiki.en.zip using wget...")
                subprocess.run(["wget", download_url, "-O", str(zip_file_path)], check=True)
                print("Download completed. Unzipping...")
                unzip_file(zip_file_path, fasttext_dir)
                print("Unzipping completed. Granting Permissions To Bin File...")
                subprocess.run(['chmod', '-R', '+x', str(bin_file_path)])
                print("Permissions granted. Continuing...")
                break
            elif user_input == 'no':
                print("Exiting the program.")
                exit(0)
            else:
                print("Please enter 'yes' or 'no'.")

def to_path(path : str):
    return os.path.expanduser(path)

def values_given(configuration: dict, parameter: str) -> bool:
    """Values for requested parameters have been supplied by the user in the configuration file

    Args:
        configuration (dict): Configuration File
        parameter (str): Requested parameter name

    Returns:
        bool: Values for requested parameter supplied
    """
    return (parameter in configuration) and (isinstance(configuration[parameter], list)) and (len(configuration[parameter]) > 0)

def get_multiples(num : int, n : int) -> list:
    """Returns a list of multiples of the requested number up to n * number

    Args:
        num (int): Number
        n (int): Multiplier

    Returns:
        list: Multiplies of num up to n * num 
    """
    multiples = []
    for i in range(1, n+1):
        multiples.append(num * i)
    return multiples

def necessary_dfs_supplied(configuration : dict) -> bool:
    """Configuration file contains values for source, target and ground truth dataframes

    Args:
        configuration (dict): Configuration file

    Raises:
        ValueError: Zero values supplied for one or more paths

    Returns:
        bool: Source, target and ground truth dataframes paths supplied within configuration dict
    """
    for path in ['source_dataset_path', 'target_dataset_path', 'ground_truth_path']:
        if(not values_given(configuration, path)):
            raise ValueError(f"{path}: No values given")
        
    return len(configuration['source_dataset_path']) == len(configuration['target_dataset_path']) == len(configuration['ground_truth_path'])

def clear_json_file(path : str):
    if os.path.exists(path):
        if os.path.getsize(path) > 0:
            open(path, 'w').close()

def purge_id_column(columns : list):
    """Return column names without the identifier column

    Args:
        columns (list): List of column names

    Returns:
        list: List of column names except the identifier column
    """
    non_id_columns : list = []
    for column in columns:
        if(column != 'id'):
            non_id_columns.append(column)
    
    return non_id_columns

def get_deepblocker_candidates(source_dataset : pd.DataFrame,
                               target_dataset : pd.DataFrame,
                               nearest_neighbors : int = 5,
                               columns_to_block : list = ["aggregate value"]
                               ) -> pd.DataFrame:
    """Applies DeepBlocker matching and retrieves the nearest neighbors 
       for each entity of the source dataset and their corresponding scores
       in a dataframe

    Args:
        columns (list): List of column names

    Returns:
        list: List of column names except the identifier column
    """        
    tuple_embedding_model = AutoEncoderTupleEmbedding()
    topK_vector_pairing_model = ExactTopKVectorPairing(K=nearest_neighbors)
    db = DeepBlocker(tuple_embedding_model, topK_vector_pairing_model)
    candidate_pairs = db.block_datasets(source_dataset, target_dataset, columns_to_block)
    return candidate_pairs

def iteration_normalized(value, iterations):
    return float(value) / float(iterations)

def update_workflow_statistics(statistics : dict,
                                candidates : pd.DataFrame,
                                ground_truth : pd.DataFrame,
                                iterations : int,
                                duplicate_of : dict
                               ) -> None:
    """Sorts the candidate pairs in descending score order globally.
       Iterates over those pairs within specified budget, 
       and updates the statistics (e.x. AUC score) for the given experiment

    Args:
        statistics (dict): Dictionary storing the statistics of the given experiment
        candidates (pd.DataFrame): Candidate pairs with their scores
        iterations (int): The number of times current workflow will be executed 
        duplicate_of (dict): Mapping from source dataset entity ID to target dataset true positive entity ID 
    """
    candidates : pd.DataFrame = candidates.sort_values(by='similarity', ascending=False)
    
    is_true_positive = candidates.apply(lambda row: row['rtable_id'] in duplicate_of[row['ltable_id']], axis=1)
    is_true_positive : list = is_true_positive.tolist()
    pairs = list(zip(candidates['ltable_id'], candidates['rtable_id']))
    pairs_info = list(zip(pairs, is_true_positive))
    seen_pairs = set()
    total_tps : int = len(ground_truth) 
    tps_found : int = 0
    total_emissions : int = 0
    recall_axis : list = []  
    tp_indices : list = []  
    budget : int = statistics['budget']  
    recall : float = 0.0  
    statistics['total_candidates'] += iteration_normalized(value=min(len(is_true_positive), budget),
                                                           iterations=iterations)

    for emission, pair_info in enumerate(pairs_info):
        if(emission >= budget or tps_found >= total_tps): 
            break
        pair, tp_emission = pair_info
        if(pair in seen_pairs):
            continue
        if(tp_emission):
            recall = (tps_found + 1.0) / total_tps
            tps_found += 1
            tp_indices.append(emission)
        recall_axis.append(recall)
        total_emissions += 1
        seen_pairs.add(pair)
     
    statistics['total_emissions'] += iteration_normalized(value=total_emissions,
                                                          iterations=iterations)
    auc : float = sum(recall_axis) / (total_emissions + 1.0)     
    statistics['auc'] += iteration_normalized(value=auc, iterations=iterations) 
    statistics['recall'] += iteration_normalized(value=recall_axis[-1], iterations=iterations)
    statistics['tp_indices'] = tp_indices
    
def gt_to_df(ground_truth : pd.DataFrame):  
    duplicate_of = defaultdict(set)
    for _, row in ground_truth.iterrows():
        id1, id2 = (row[0], row[1])
        if id1 in duplicate_of: duplicate_of[id1].add(id2)
        else: duplicate_of[id1] = {id2}
    return duplicate_of


def get_valid_indexings(indexing : str):
    if(indexing == "inorder"):
        return ["inorder"]
    elif(indexing == "reverse"):
        return ["reverse"]
    else:
        return ["inorder", "reverse"]
    
def get_candidates_based_on_indexing(indexing : str, inorder_candidates : pd.DataFrame, reverse_candidates : pd.DataFrame):
    if(indexing == "inorder"):
        return inorder_candidates
    if(indexing == "reverse"):
        return reverse_candidates
    else:
        return pd.concat([inorder_candidates, reverse_candidates], axis=0)
