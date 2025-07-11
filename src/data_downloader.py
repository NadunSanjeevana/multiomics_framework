import os
import pandas as pd
import requests
import scanpy as sc
import gdown

# Create a folder for the data
os.makedirs("data", exist_ok=True)

# 1. Transcriptomics Data ----------------------------------------------------
# GTEx TPMs
print("Downloading GTEx transcriptomics data...")
gtex_url = "https://storage.googleapis.com/gtex_analysis_v8/rna_seq_data/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz"
gtex_df = pd.read_csv(gtex_url, sep='\t', skiprows=2)
gtex_df.to_csv("data/gtex_tpm.csv", index=False)
print("Saved: data/gtex_tpm.csv")

# HCA - PBMC 3k example using scanpy
print("Downloading Human Cell Atlas PBMC3k dataset...")
adata = sc.datasets.pbmc3k()
adata.to_df().to_csv("data/hca_pbmc3k.csv")
print("Saved: data/hca_pbmc3k.csv")

# 2. Proteomics Data --------------------------------------------------------
# Human Protein Atlas
print("Downloading Human Protein Atlas protein expression data...")
hpa_url = "https://www.proteinatlas.org/download/proteinatlas.tsv.zip"
hpa_df = pd.read_csv(hpa_url, sep='\t')
hpa_df.to_csv("data/hpa_protein_expression.csv", index=False)
print("Saved: data/hpa_protein_expression.csv")

# 3. Epigenomics Metadata (ENCODE) -----------------------------------------
print("Querying ENCODE ATAC-seq metadata...")
encode_query = "https://www.encodeproject.org/search/?type=Experiment&assay_title=ATAC-seq&biosample_ontology.organ_slims=brain&status=released&format=json"
res = requests.get(encode_query, headers={"accept": "application/json"})
hits = res.json()['@graph']
metadata = []
for hit in hits[:20]:
    metadata.append({
        "accession": hit.get('accession', ''),
        "biosample": hit.get('biosample_ontology', {}).get('term_name', ''),
        "target": hit.get('target', {}).get('label', ''),
        "description": hit.get('description', ''),
        "url": f"https://www.encodeproject.org{hit.get('@id', '')}"
    })
meta_df = pd.DataFrame(metadata)
meta_df.to_csv("data/encode_atac_metadata.csv", index=False)
print("Saved: data/encode_atac_metadata.csv")

# 4. Metabolomics Data (HMDB) ----------------------------------------------
print("Downloading HMDB metabolomics data (sample file)...")
hmdb_url = "https://www.dropbox.com/s/q1v8z0olrpr3s3k/hmdb_metabolites_concentration.csv?dl=1"
gdown.download(hmdb_url, "data/hmdb_metabolites_concentration.csv", quiet=False)
print("Saved: data/hmdb_metabolites_concentration.csv")

# 5. Annotation Data --------------------------------------------------------
# Gene Ontology Annotations
print("Downloading Gene Ontology (GO) annotations...")
go_url = "http://current.geneontology.org/annotations/goa_human.gaf.gz"
go_df = pd.read_csv(go_url, sep='\t', comment='!', header=None)
go_df.to_csv("data/goa_human.csv", index=False)
print("Saved: data/goa_human.csv")

# KEGG Human Pathways using bioservices
print("Downloading KEGG human pathway list...")
from bioservices import KEGG
kegg = KEGG()
kegg_result = kegg.list("pathway", "hsa")
with open("data/kegg_human_pathways.txt", "w") as f:
    f.write(kegg_result)
print("Saved: data/kegg_human_pathways.txt")

print("\n✅ All data downloaded successfully into 'data/' folder.")
