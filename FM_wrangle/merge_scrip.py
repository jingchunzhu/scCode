#!/usr/bin/env python
# coding: utf-8

import pandas as pd

obs_file_1 = "benchmark_results/zilan_ref_Wang2025Nature_test_53ce2631-3646-4172-bbd9-38b0a44d8214_k30/53ce2631-3646-4172-bbd9-38b0a44d8214_knn_10_top2_labels_scores.tsv"
obs_file_2 = "benchmark_results/ref_Wang2025Nature_test_53ce2631-3646-4172-bbd9-38b0a44d8214_k10/predictions.tsv"
obs_file_3 = "benchmark_results/ref_Wang2025Nature_test_53ce2631-3646-4172-bbd9-38b0a44d8214_k10/ic/per_cell_evaluation.tsv"

output = 'merge_prediction.tsv'

df1 = pd.read_csv(obs_file_1, sep='\t') #zilan
df2 = pd.read_csv(obs_file_2, sep='\t', comment="#")
df3 = pd.read_csv(obs_file_3, sep='\t', comment="#")
assert(len(df1) == len(df2) == len(df3))

# Drop columns from df2 that already exist in df1
dup_cols_2 = [c for c in df2.columns if c in df1.columns]
df2 = df2.drop(columns=dup_cols_2)

# Drop columns from df3 that already exist in df1 or df2
existing_cols = set(df1.columns) | set(df2.columns)
dup_cols_3 = [c for c in df3.columns if c in existing_cols]
df3 = df3.drop(columns=dup_cols_3)

result = pd.concat([df1, df2, df3], axis=1)

# Set 'cell_id' as index
result = result.set_index('cell_id')

# Then export (index=True to include it, or False to exclude)
result.to_csv(output, sep='\t', index=True)

