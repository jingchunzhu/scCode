import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--embedding",
    required=True,
    choices=["uce", "scimilarity"],
    help="Embedding type to use (uce or scimilarity)"
)

parser.add_argument(
    "--results_dir",
    required=True,
    help="Directory path where the output results will be saved"
)

args = parser.parse_args()
embedding = args.embedding
results_dir = args.results_dir

import cellxgene_census  # python3.11 
import os, sys, re
import numpy as np
import pandas as pd
import h5py

obs_only = False
uce_only = False

if embedding =="uce":
    embedding_outfile = "uce.h5"
    data_dirs =["UCE_npy_obs"]
    n_cols = 1280
elif embedding == "scimilarity":
    embedding_outfile = "scimilarity.h5"
    data_dirs =["SCimilarity_npy_obs"]
    n_cols = 128
else:
    print ("embedding method not supported yet")
    sys.exit(1)
os.makedirs(results_dir, exist_ok=True)

select_datasets = []
file_path = os.path.join(results_dir, "dataset_ids.tsv")
if os.path.exists(file_path):
    with open(file_path, 'r') as f:
        select_datasets = [line.strip() for line in f if line.strip()]

obs_outfile = "obs.tsv.gz"

ucsc_file = "datasetLists/ucsc_datasets.txt"
cgCensus_non_primary_file ="datasetLists/cgCensus_non_primary_datasets.txt"
ignore_file = "datasetLists/ignore_datasets.txt"

dataset_list = []
obs_list =[]
cellCount = 0
arr_dtype = np.dtype('float32')
counter = 0
select_cols = ['disease', 'disease_ontology_term_id',
               'tissue', 'tissue_ontology_term_id',
               'cell_type', 'cell_type_ontology_term_id',
               'assay', 'assay_ontology_term_id',
               'development_stage', 'development_stage_ontology_term_id',
               "organism","organism_ontology_term_id",
               "sex","sex_ontology_term_id",
               "self_reported_ethnicity","self_reported_ethnicity_ontology_term_id",
               'donor_id',
               'sample_id',
               'suspension_type',
               'tissue_type',
               'dataset_id',
               'is_primary_data',
               ]

def get_ucsc_dataset_ids():
    datasets =[]
    # Read dataset IDs from the text file (one per line)
    with open(ucsc_file, "r") as f:
        dataset_info = [line.strip().split("\t") for line in f if (line.strip() and line[0] != "#")]
    for item in dataset_info:
        datasets.append(item[0])
    return datasets

def get_cgCensus_non_primary_dataset_ids():
    datasets =[]
    # Read dataset IDs from the text file (one per line)
    with open(cgCensus_non_primary_file, "r") as f:
        dataset_info = [line.strip().split("\t") for line in f if (line.strip() and line[0] != "#")]
    for item in dataset_info:
        datasets.append(item[0])
    return datasets

def get_ignore_dataset_ids():
    datasets =[]
    # Read dataset IDs from the text file (one per line)
    with open(ignore_file, "r") as f:
        dataset_info = [line.strip().split("\t") for line in f if (line.strip() and line[0] != "#")]
    for item in dataset_info:
        datasets.append(item[0])
    return datasets


def get_ucsc_collection():
    df = pd.read_csv(
        ucsc_file,
        sep="\t",
        comment="#",      # ignore any line starting with '#'
        index_col=0       # use the first column as index
    )
    return df

#cellxgene info
latest_census = cellxgene_census.open_soma(census_version = "latest")
cxgInfo = latest_census["census_info"]["datasets"].read().concat().to_pandas()

#ucsc info
ucscInfo = get_ucsc_collection()

# file sets
ucsc_datasets = get_ucsc_dataset_ids()
cgCensus_non_primary_datasets = get_cgCensus_non_primary_dataset_ids()
ignore_datasets = get_ignore_dataset_ids()

if not obs_only:
    fuce = h5py.File(os.path.join(results_dir, embedding_outfile), "w")
    dset_uce = fuce.create_dataset("data",
                                   shape=(0, n_cols),
                                   maxshape=(None, n_cols),
                                   dtype=arr_dtype
                                   )

for singleDir in data_dirs:
    for root, dirs, files in os.walk(singleDir):
        for file in files:
            #if counter >1:
            #    break
            if file.endswith('.npy'):
                dataset_id = file.removesuffix("_uce.npy").removesuffix("_scimilarity.npy")
                match = re.match(r'^((?:ucsc-)?[a-f0-9\-]+)_', file)
                real_dataset_id = match.group(1)      

                if len(select_datasets) !=0 and real_dataset_id not in select_datasets:
                    continue
                if len(select_datasets) ==0 and real_dataset_id in ignore_datasets:
                    continue
                
                print (singleDir, dataset_id)
                if dataset_id not in dataset_list:
                    dataset_list.append(dataset_id)

                    uce_filepath = os.path.join(singleDir, file)
                    uce  = np.load(uce_filepath)
                    obs_filepath = os.path.join(singleDir, dataset_id+"_obs.tsv")
                    obs = pd.read_csv(obs_filepath, sep='\t', low_memory=False)

                    # First condition: no NaNs in the row
                    mask_uce = ~np.isnan(uce).any(axis=1) # Keep rows where no NaN is present
                    # Second condition: 'is_primary_data' column is True
                    if "is_primary_data" in obs.columns and \
                       dataset_id not in ucsc_datasets and \
                       dataset_id not in cgCensus_non_primary_datasets :
                        mask_primary = (obs["is_primary_data"] == True).values
                    else:
                        mask_primary = np.ones(len(obs), dtype=bool)  # Keep all rows if column doesn't exist
                    
                    # Combine the two condition masks
                    combined_mask = mask_uce & mask_primary
                    if combined_mask.sum() > 1:
                        counter = counter + 1
                        uce = uce[combined_mask]
                        obs = obs[combined_mask]

                        cellCount = cellCount + uce.shape[0]
                        print (counter, "total cell count:", cellCount)

                        # handle obs: index
                        if not uce_only:
                            obs['new_index'] = obs['dataset_id'].astype(str) + "_" + obs.index.astype(str)
                            obs.rename(columns={"Unnamed: 0": "old_index"}, inplace=True)
                            obs.set_index('new_index', inplace=True)
                            obs = obs.assign(dataset_id = real_dataset_id)
                            # modify ucsc dataset cell_type and cell_type_terminoloyg_id
                            if dataset_id in ucsc_datasets:
                                if "cell_type" in ucscInfo.columns:
                                    column =  ucscInfo.at[dataset_id, "cell_type"]
                                    obs["cell_type"] = obs[column]
                                if "cell_type_ontology_term_id" in ucscInfo.columns:
                                    column =  ucscInfo.at[dataset_id, "cell_type_ontology_term_id"]
                                    obs["cell_type_ontology_term_id"] = obs[column]
                            if len(select_datasets) ==0: # for combining all datasets, have to select columns
                                cols_to_use = [c for c in select_cols if c in obs.columns]
                                obs = obs[cols_to_use]
                            # add collection info
                            if dataset_id not in ucsc_datasets:
                                obs["collection_doi_label"] = cxgInfo[cxgInfo.dataset_id == dataset_id]["collection_doi_label"].values[0]
                                obs["collection_doi"] = cxgInfo[cxgInfo.dataset_id == dataset_id]["collection_doi"].values[0]
                                obs["dataset_title"] = cxgInfo[cxgInfo.dataset_id == dataset_id]["dataset_title"].values[0]
                            else:
                                obs["collection_doi"] = ucscInfo.at[dataset_id, "collection_doi"]
                                obs["collection_doi_label"] = ucscInfo.at[dataset_id, "collection_doi_label"]
                                obs["dataset_title"] = ucscInfo.at[dataset_id, "dataset_title"]
                            obs_list.append(obs)
                                                    
                        ## export to dest_uce, one numpy array at a time concatenate at the end, memory efficient and fast
                        if not obs_only:
                            dset_uce.resize(dset_uce.shape[0] + uce.shape[0], axis=0)
                            dset_uce[-uce.shape[0]:] = uce
                        
                    else:
                        print ("All embedding data are NaN (very rare) or is_primary_data are all False (more likely)")
                else:
                    print ("Duplicate dataset")
if not obs_only:
    fuce.close()

if not uce_only:
    print("Start with obs processing")
    combined_obs = pd.concat(obs_list, axis=0, join='outer')

    ## shape
    num_cols = combined_obs.shape[1]

    print ("combined obs shape:", (cellCount, num_cols))

    #export combined_obs
    combined_obs.to_csv(os.path.join(results_dir, obs_outfile), sep='\t')

print(f"these {len(dataset_list)} datasets are combined")
print (dataset_list)
