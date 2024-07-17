# Progressive Entity Resolution - A Design Space Exploration

This repository consists of all the necessary codes and imports to reproduce the numerical and graphical
results of the ***Progressive entity resolution: a design space exploration*** paper. The included scripts
allow for quick automation and deployment of custom PER workflow setups from scratch.

## Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [Workflow Setup Execution](#workflow-setup-execution)
- [DeepBlocker Setup Execution](#deepblocker-setup-execution)
- [Sparkly Setup Execution](#sparkly-setup-execution)

## Installation

Each **PER algorithm category** — *classic*, *deepblocker*, and *sparkly* — requires its own initialization procedure and installation steps.
Before running any of these algorithms, you must first download the basic Python requirements that are shared across all setups.

### Steps to Initialize

**1. Install Shared Requirements:**

Start by cloning this repository:
```bash
git clone <repository-web-link>
```
Enter the repository and install the basic Python packages needed for all algorithms:
```bash
cd PER-Design-Space-Exploration/ 
pip install -r requirements.txt
```
**2. DeepBlocker Requirements:**

DeepBlocker scripts allow the user to install fasttext embeddings on the go. Yet, we highly suggest doing that manually.
The downloaded files will be identified by the program and the configuration will be executed on the spot.

Start by creating a ```fasttext``` directory at root:
```bash
cd ~
mkdir fasttext
cd fasttext
```
 Download the embeddings, unzip them and grant proper permissions:
```bash
wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip
tar -xzvf wiki.en.zip
chmod +x wiki.en.bin
```

**3. Sparkly Requirements:**

First, you will have to download ```pylucene```. From the repository path:
```bash
wget https://dlcdn.apache.org/lucene/pylucene/pylucene-9.7.0-src.tar.gz
tar -xzvf pylucene-9.7.0-src.tar.gz 
```
In ```Makefile``` uncomment the code snippet corresponding to your operating system.
Replace it with the following:
```makefile
# Linux     (Debian Bullseye 64-bit, Python 3.9.2, Temurin Java 17
PREFIX_PYTHON=/usr
PYTHON=$(PREFIX_PYTHON)/local/bin/python
JCC=$(PYTHON) -m jcc --shared
NUM_FILES=16
```
Create the Sparkly Docker image:
```bash
docker build -t sparkly . 
```

## Configuration

Setup is defined as a set of viable PER algorithm definition groups. We associate each group with a key (most commonly a dataset identifier) and a set of values for each argument. Examples of such setups can be found in ```grid_config```, where we define the best-performing parameter combinations for each PER algorithm category (a separate file for each top-K combination and category). Let's analyze the best workflow for the LLM approach for D1:
```json
{
    "d1": {
        "matcher": ["EmbeddingsNNBPM"],
        "algorithm": ["BFS"],
        "budget" : [],
        "dataset_name" : ["D1"],
        "source_dataset_path": ["~/datasets/D1/rest1.csv"],
        "target_dataset_path": ["~/datasets/D1/rest2.csv"],
        "ground_truth_path": ["~/datasets/D1/gt.csv"],
        "number_of_nearest_neighbors": [5],
        "indexing": ["bilateral"],
        "similarity_function": ["euclidean"],
        "language_model": ["st5"],
        "separator": ["|"],
        "tokenizer": [],
        "weighting_scheme": [],
        "window_size": [],
        "iterations": [10]
    }
}
```
In this case, we defined a single value for each of the parameters of a ```EmbeddingsNNBPM``` (Vector-Based Block Building) configuration, which is assigned the key ```d1``` implying it will be applied to D1. 
We also defined the paths, where the dataframes for D1 are stored and the separator used within them. Each parameter is pointing to a list, meaning we can assign multiple values to them. The final configuration is
defined as the combination of all the values that we have assigned. Finally, we can also define the number of iterations over which we want to run all the combinations, in which case the performance metrics are averaged. 
Some of the parameters can be left empty. They are either irrelevant to our configuration or are auto-completed within the scripts
(e.x. ```budget``` which when left unassigned is equivalent to a list of multipliers of the size of the ground-truth dataframe. 

Each configuration category has a different set of valid parameters. That is why we highly recommend you have an extensive look at the best configuration definitions in ```grid_config```.

## Workflow Setup Execution

We allow for 4 basic workflow setups. Each one of them can contain more than one matcher:
* NN Based  (```EmbeddingsNNBPM```)
* Block Based {```PESM```)
* Sorted Neighborhoods (```GlobalPSNM```, ```LocalPSNM```)
* Joins (```TopKJoinPM```)

Suppose we want to execute the ***best (top-1) Progressive Top-K Join*** configuration for D3.
The configuration for said dataset has been mapped to key ```"d3"``` in the top-1 Joins setup file `./grid_config/join_experiments_top_1.json`.
To execute said configuration in the background, we simply run:
```bash
nohup python3 run_workflow_setup.py
--config_path ./grid-configs/pesm_experiments_top_1.json
--dataset d3
> ./outs/top_1_joins_d3.out &
```
Don't forget to create an outs folder for your log files!
The basic workflow configuration script has the following format:
```bash
python3 run_workflow_setup.py
--config_path <absolute path to the setup file>
--store_folder_path <absolute path to stored results>
--dataset <configuration key within setup file>
--building True/False <construct blocks on initialization>
--filtering True/False <filter blocks>
--purging True/False <purge blocks>
```
## DeepBlocker Setup Execution

Can be executed similarly to a basic workflow. That is the format:
```bash
python3 run_deepblocker_setup.py
--config_path <absolute path to the setup file>
--store_folder_path <absolute path to stored results>
--dataset <configuration key within setup file>
```

## Sparkly Setup Execution
Before proceeding, make sure you have properly initialized a Sparkly Image as described in [Installation](#installation).
Suppose we want to execute the ***best Sparkly configuration*** for D4. As the configuration is already in the setup
file ```grid_config/sparkly_experiments_top_1.json``` (within the default configuration folder) and mapped to key ```d4```,
we can execute it with the following command:

```bash
./run_sparkly_setup.sh d4 sparkly_experiments_top_1
```
Make sure the bash script has the proper permissions!
The Sparkly script takes 5 parameters out of which the first 2 are mandatory:
```bash
./run_sparkly_setup.sh
<Configuration key within the setup file> - Mandatory
<Name of file containing target configuration> - Mandatory
<Name of the folder containing configuration setup file> - Optional [default - grid_config]
<Name of the folder where results will be stored> - Optional [default - results]
<Path from which we reference everything> - Optional [default - execution path]
```

