# src/preprocessor.py
"""
Multi-Omics Data Preprocessor Module
Handles normalization and standardization of different omics data types
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
import warnings
warnings.filterwarnings('ignore')

class MultiOmicsPreprocessor:
    """Preprocessor for multi-omics datasets"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.scalers = {}
        self.imputers = {}
        
    def preprocess_all(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Preprocess all datasets"""
        processed_datasets = {}
        
        for data_type, data in datasets.items():
            if data_type == 'annotations':
                processed_datasets[data_type] = data
                continue
                
            self.logger.info(f"Preprocessing {data_type} data...")
            processed_datasets[data_type] = self._preprocess_dataset(data, data_type)
            
        return processed_datasets
    
    def _preprocess_dataset(self, data: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Preprocess individual dataset"""
        # Handle missing values
        data = self._handle_missing_values(data, data_type)
        
        # Apply normalization
        data = self._normalize_data(data, data_type)
        
        # Remove low-variance features
        data = self._remove_low_variance_features(data, data_type)
        
        # Log transformation for certain data types
        if data_type in ['transcriptomics', 'metabolomics']:
            data = self._log_transform(data)
        
        # Final imputation to ensure no NaNs remain
        data = data.fillna(0)
        self.logger.info(f"Final fillna(0) applied for {data_type} to ensure no NaNs remain.")
        
        return data
    
    def _handle_missing_values(self, data: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Handle missing values based on data type"""
        missing_strategy = self.config.get('missing_value_strategy', {}).get(data_type, 'mean')
        
        if missing_strategy == 'knn':
            imputer = KNNImputer(n_neighbors=5)
            self.imputers[data_type] = imputer
            data_imputed = imputer.fit_transform(data.T).T
            data = pd.DataFrame(data_imputed, index=data.index, columns=data.columns)
        else:
            imputer = SimpleImputer(strategy=missing_strategy)
            self.imputers[data_type] = imputer
            data_imputed = imputer.fit_transform(data.T).T
            data = pd.DataFrame(data_imputed, index=data.index, columns=data.columns)
        
        self.logger.info(f"Handled missing values for {data_type} using {missing_strategy}")
        return data
    
    def _normalize_data(self, data: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Normalize data based on type"""
        norm_method = self.config.get('normalization', {}).get(data_type, 'z-score')
        
        if norm_method == 'z-score':
            scaler = StandardScaler()
        elif norm_method == 'robust':
            scaler = RobustScaler()
        else:
            return data
        
        self.scalers[data_type] = scaler
        data_normalized = scaler.fit_transform(data.T).T
        data = pd.DataFrame(data_normalized, index=data.index, columns=data.columns)
        
        self.logger.info(f"Normalized {data_type} data using {norm_method}")
        return data
    
    def _remove_low_variance_features(self, data: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Remove features with low variance"""
        variance_threshold = self.config.get('variance_threshold', {}).get(data_type, 0.01)
        
        # Calculate variance across samples
        variances = data.var(axis=1)
        high_var_features = variances[variances > variance_threshold].index
        
        data_filtered = data.loc[high_var_features]
        
        removed_features = len(data) - len(data_filtered)
        self.logger.info(f"Removed {removed_features} low-variance features from {data_type}")
        
        return data_filtered
    
    def _log_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply log transformation"""
        # Add small constant to avoid log(0)
        data_log = np.log2(data + 1)
        self.logger.info("Applied log2 transformation")
        return data_log
    
    def _apply_tpm_normalization(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply TPM normalization for RNA-seq data"""
        # Assuming gene lengths are provided or use median gene length
        gene_lengths = np.random.uniform(1000, 5000, size=len(data))
        
        # Calculate RPK (reads per kilobase)
        rpk = data.div(gene_lengths, axis=0) * 1000
        
        # Calculate scaling factor
        scaling_factor = rpk.sum(axis=0) / 1e6
        
        # Calculate TPM
        tpm = rpk.div(scaling_factor, axis=1)
        
        self.logger.info("Applied TPM normalization")
        return tpm
    
    def transform_new_data(self, data: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Transform new data using fitted preprocessors"""
        # Handle missing values
        if data_type in self.imputers:
            data_imputed = self.imputers[data_type].transform(data.T).T
            data = pd.DataFrame(data_imputed, index=data.index, columns=data.columns)
        
        # Apply normalization
        if data_type in self.scalers:
            data_normalized = self.scalers[data_type].transform(data.T).T
            data = pd.DataFrame(data_normalized, index=data.index, columns=data.columns)
        
        # Apply log transformation if needed
        if data_type in ['transcriptomics', 'metabolomics']:
            data = np.log2(data + 1)
        
        return data