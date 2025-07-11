# src/data_loader.py
"""
Multi-Omics Data Loader Module
Handles loading data from various omics sources
"""

import pandas as pd
import numpy as np
import requests
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import gzip
import json
import os

class MultiOmicsDataLoader:
    """Loader for multi-omics datasets"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load all configured datasets"""
        datasets = {}
        
        # Load transcriptomics data
        if self.config.get('transcriptomics', {}).get('enabled', False):
            datasets['transcriptomics'] = self._load_transcriptomics()
            
        # Load proteomics data
        if self.config.get('proteomics', {}).get('enabled', False):
            datasets['proteomics'] = self._load_proteomics()
            
        # Load epigenomics data
        if self.config.get('epigenomics', {}).get('enabled', False):
            datasets['epigenomics'] = self._load_epigenomics()
            
        # Load metabolomics data
        if self.config.get('metabolomics', {}).get('enabled', False):
            datasets['metabolomics'] = self._load_metabolomics()
            
        # Load annotation data
        datasets['annotations'] = self._load_annotations()
        
        return datasets
    
    def _load_transcriptomics(self) -> pd.DataFrame:
        """Load transcriptomics data from GTEx (real data)"""
        file_path = "data/gtex_tpm.csv"
        self.logger.info(f"Loading transcriptomics data from {file_path} ...")
        if not os.path.exists(file_path):
            self.logger.warning(f"Transcriptomics file not found: {file_path}")
            return pd.DataFrame()
        df = pd.read_csv(file_path, sep=",", low_memory=False)
        self.logger.info(f"Loaded transcriptomics data: {df.shape}")
        return df
    
    def _load_proteomics(self) -> pd.DataFrame:
        """Load proteomics data from Human Protein Atlas (real data)"""
        file_path = "data/hpa_protein_expression.csv"
        self.logger.info(f"Loading proteomics data from {file_path} ...")
        if not os.path.exists(file_path):
            self.logger.warning(f"Proteomics file not found: {file_path}")
            return pd.DataFrame()
        df = pd.read_csv(file_path, sep=",", low_memory=False)
        self.logger.info(f"Loaded proteomics data: {df.shape}")
        return df
    
    def _load_epigenomics(self) -> pd.DataFrame:
        """Load epigenomics metadata from ENCODE (real data)"""
        file_path = "data/encode_atac_metadata.csv"
        self.logger.info(f"Loading epigenomics metadata from {file_path} ...")
        if not os.path.exists(file_path):
            self.logger.warning(f"Epigenomics file not found: {file_path}")
            return pd.DataFrame()
        df = pd.read_csv(file_path, sep=",", low_memory=False)
        self.logger.info(f"Loaded epigenomics metadata: {df.shape}")
        return df
    
    def _load_metabolomics(self) -> pd.DataFrame:
        """Load metabolomics data from HMDB (real data)"""
        file_path = "data/hmdb_metabolites_concentration.csv"
        self.logger.info(f"Loading metabolomics data from {file_path} ...")
        if not os.path.exists(file_path):
            self.logger.warning(f"Metabolomics file not found: {file_path}")
            return pd.DataFrame()
        df = pd.read_csv(file_path, sep=",", low_memory=False)
        self.logger.info(f"Loaded metabolomics data: {df.shape}")
        return df

    def _load_annotations(self) -> Dict[str, pd.DataFrame]:
        """Load annotation data from real files (GO, KEGG, etc.)"""
        self.logger.info("Loading annotation data from real files...")
        annotations = {}
        go_path = "data/goa_human.csv"
        if os.path.exists(go_path):
            annotations['go'] = pd.read_csv(go_path, sep=",", low_memory=False)
        else:
            self.logger.warning(f"GO annotation file not found: {go_path}")
            annotations['go'] = pd.DataFrame()
        # KEGG pathways: if you have a CSV, load it; otherwise, skip or parse txt
        # pathways_path = "data/kegg_human_pathways.csv"
        # if os.path.exists(pathways_path):
        #     annotations['pathways'] = pd.read_csv(pathways_path, sep=",", low_memory=False)
        # else:
        #     self.logger.warning(f"KEGG pathways file not found: {pathways_path}")
        #     annotations['pathways'] = pd.DataFrame()
        return annotations