# Multi-Omics Integration Pipeline

This repository implements a multi-omics integration pipeline for disease gene prioritization and drug repurposing. The pipeline integrates transcriptomics, proteomics, epigenomics, and metabolomics data, applies advanced feature engineering, and scores drugs using a weighted aggregation strategy.

## Features
- Acquisition and integration of multi-omics datasets
- Preprocessing and normalization (TPM, z-score, etc.)
- Mapping to unified gene identifiers (HGNC)
- Feature engineering with pathway, PPI, and annotation enrichment
- Dimensionality reduction (PCA)
- Multi-view data fusion (CCA/MKL)
- Weighted drug scoring

## Requirements
Install dependencies with:

```bash
pip install -r requirements.txt
```

## Configuration
Create a configuration file at `config/config.yaml` (see example below):

```yaml
data:
  transcriptomics:
    enabled: true
  proteomics:
    enabled: true
  epigenomics:
    enabled: true
  metabolomics:
    enabled: true
preprocessing:
  normalization:
    transcriptomics: tpm
    proteomics: z-score
    epigenomics: z-score
    metabolomics: z-score
  missing_value_strategy:
    transcriptomics: mean
    proteomics: mean
    epigenomics: mean
    metabolomics: mean
  variance_threshold:
    transcriptomics: 0.01
    proteomics: 0.01
    epigenomics: 0.01
    metabolomics: 0.01
feature_engineering:
  pca_components:
    transcriptomics: 50
    proteomics: 30
    epigenomics: 20
    metabolomics: 10
model:
  # Model parameters here
drug_scoring:
  # Drug scoring parameters here
```

## Usage

Run the pipeline:

```bash
python src/main.py --config config/config.yaml --mode full --output_dir results
```

- `--mode` can be `train`, `predict`, or `full`.
- Results will be saved in the specified output directory.

## Output
- Trained model: `results/model.pkl`
- Features: `results/features.pkl`
- Drug scores: `results/drug_scores.csv`

## Notes
- The current implementation uses synthetic/demo data. Replace data loading methods with real data sources as needed.
- Ensure you have a valid configuration file before running the pipeline. 