## the code to generate cellxgene datasets information file
## file: datasetLists/datasets.info.cxg.{date}.txt

import cellxgene_census ##python 3.11 kernel
import pandas as pd
import datetime

latest_census = cellxgene_census.open_soma(census_version = "latest")
#stable_census = cellxgene_census.open_soma(census_version = "stable")

latest_census_datasets = latest_census["census_info"]["datasets"].read().concat().to_pandas()
df = latest_census_datasets[["dataset_id", "dataset_total_cell_count", "collection_doi", "collection_doi_label", "dataset_title" ]]


today = datetime.date.today()
output = f"datasets.info.cxg.{today}.txt"
with open(output, "w") as f:
    f.write("#")
    df.to_csv(f, sep="\t", index=False)
    

