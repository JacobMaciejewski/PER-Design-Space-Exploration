import os
import argparse
from pyjedai.visualization import plot_attributes_performance_for_budget

def valid_file(file : str):
    return ('experiments' in file and '.csv' in file)

def get_attributes(file : str):
    if 'pesm' in file or 'gt' in file or 'test' in file:
        return ["weighting_scheme", "algorithm"]
    elif 'gsn' in file or 'lsn' in file:
        return ["weighting_scheme", "window_size", "algorithm"]
    elif 'nn' in file:
        return ["number_of_nearest_neighbors", "language_model", "similarity_function"]
    elif 'join' in file:
        return ["number_of_nearest_neighbors", "similarity_function", "tokenizer", "weighting_scheme", "qgram"]
    else:
        return ["indexing"]
    
def get_method_csv_files_for_directory(method : str, directory_path: str) -> list:
    file_names = os.listdir(directory_path)
    return sorted([os.path.join(directory_path, file_name) for file_name in file_names if (valid_file(file_name) and method in file_name)])

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments_path', type=str, required=True,
                        help='Path should include the workflow results per dataset file')
    parser.add_argument('--features', type=str, nargs='+', default=['auc', 'recall', 'time'],
                        help='List of target metrics')
    parser.add_argument('--calculate_distance', type=str, nargs='+', default=['True', 'False'],
                        help='Calculate Distances')
    args = parser.parse_args()
    
    # map string representation of true/false into boolean
    calculate_distances = [option.lower() == 'true' for option in args.calculate_distance]

    # first attribute (seperated by underscore) of a experiments result file corresponds to the method used in its calculation
    # we gather all the unique methods whose results are stored in the given folder
    file_list = os.listdir(args.experiments_path)
    worfklow_types = set([file_name.split('_')[0] for file_name in file_list if valid_file(file_name)])

    total_files = len(worfklow_types) * len(args.features) * len(args.calculate_distance)
    current_file = 0
    print(args.experiments_path)
    
    for worfklow_type in worfklow_types:
        attributes = get_attributes(worfklow_type)
        for feature in args.features:
            for calculate_distance in calculate_distances:
                current_file += 1
                metric = "distance" if calculate_distance else "ranking"
                print(f"{current_file}/{total_files} : Workflow Type[{worfklow_type}] Attributes{attributes} Feature[{feature}] Metric[{metric}]")
                plot_attributes_performance_for_budget(
                                                        method_name = worfklow_type,
                                                        feature = feature,
                                                        attributes = attributes,
                                                        load_paths = get_method_csv_files_for_directory(method=worfklow_type, directory_path=args.experiments_path),
                                                        calculate_distance=calculate_distance
                                                    )