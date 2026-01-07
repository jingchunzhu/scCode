import cellxgene_census
import pandas as pd
import numpy as np
import os, re
import argparse

census_version="2023-12-15" # Universal Cell Embeddings (UCE) run on this release
results_dir = "cgCensus2023-12-15"

def getHumanUCE(census, dataset_id):
    adata = cellxgene_census.get_anndata(
        census,
        organism = "homo_sapiens",
        #measurement_name = "RNA",
        obs_value_filter = f"dataset_id == '{dataset_id}' and is_primary_data == True",
        obs_embeddings = ["uce"]
    )    
    print(f"Number of cells: {adata.n_obs}")
    return adata

def saveUCE (adata, dataset_id):
    uce_df = pd.DataFrame(adata.obsm['uce'])
    np.save(os.path.join(results_dir, f'{dataset_id}_uce.npy'), uce_df)
    adata.obs.to_csv(os.path.join(results_dir, f'{dataset_id}_obs.tsv') , sep='\t')
    
def result_exists(dataset_id):
    if os.path.exists(os.path.join(results_dir, f'{dataset_id}_uce.npy')):
        return True
    else:
        return False
    
parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--dataset_id', type=str, help='Single dataset ID')
group.add_argument('--dataset_id_file', type=str, help='File containing list of dataset IDs (one per line)')
args = parser.parse_args()

census = cellxgene_census.open_soma(census_version= census_version) 

if args.dataset_id:
    dataset_id = args.dataset_id
    if result_exists(dataset_id):
        print (f"{dataset_id} uce results already downloaded")
    else:
        print (f"{dataset_id} downloading")
        adata = getHumanUCE(census, dataset_id)
        if adata.n_obs > 0:
            saveUCE (adata, dataset_id)
    
elif args.dataset_id_file:
    with open(args.dataset_id_file, 'r') as f:
        dataset_ids = f.read().splitlines()  # Read the list of dataset IDs
        dataset_ids = [re.sub(r'\s+', '', id_) for id_ in dataset_ids if id_.strip() and not id_.strip().startswith('#')]  # skip lines start with #, remove all whitespace charactre such /t /r etc 

    for dataset_id in dataset_ids:
        if result_exists(dataset_id):
            print (f"{dataset_id} uce results already downloaded")
        else:
            print (f"{dataset_id} downloading")
            adata = getHumanUCE(census, dataset_id)
            if adata.n_obs > 0:
                saveUCE (adata, dataset_id)

