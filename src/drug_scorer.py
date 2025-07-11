# src/drug_scorer.py
"""
Drug Scoring Module
Implements weighted aggregation strategy for drug repurposing
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Tuple
from pathlib import Path
import requests
import json

class DrugScorer:
    """Drug scoring for repurposing based on weighted aggregation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.drug_target_db = None
        
    def score_drugs(self, model_dict: Dict[str, Any], 
                   features: Dict[str, Any]) -> pd.DataFrame:
        """Score drugs based on target gene risk scores"""
        
        # Load drug-target associations
        drug_targets = self._load_drug_targets()
        
        # Get gene risk scores from model
        gene_risk_scores = self._calculate_gene_risk_scores(model_dict, features)
        
        # Calculate drug scores using weighted aggregation
        drug_scores = self._calculate_drug_scores(drug_targets, gene_risk_scores, features)
        
        # Rank drugs by score
        ranked_drugs = drug_scores.sort_values('composite_score', ascending=False)
        
        self.logger.info(f"Scored {len(ranked_drugs)} drugs")
        return ranked_drugs
    
    def _load_drug_targets(self) -> pd.DataFrame:
        """Load drug-target associations"""
        # For demo purposes, create synthetic drug-target data
        # In practice, would load from DrugBank, ChEMBL, etc.
        
        np.random.seed(42)
        
        # Generate synthetic drugs
        n_drugs = 1000
        drugs = [f"DRUG_{i:04d}" for i in range(n_drugs)]
        
        # Generate synthetic targets (genes)
        n_genes = 5000
        genes = [f"GENE_{i}" for i in range(n_genes)]
        
        # Create drug-target associations
        drug_target_pairs = []
        for drug in drugs:
            # Each drug targets 1-10 genes
            n_targets = np.random.randint(1, 11)
            targets = np.random.choice(genes, n_targets, replace=False)
            
            for target in targets:
                # Add binding affinity score
                binding_score = np.random.uniform(0.3, 1.0)
                drug_target_pairs.append({
                    'drug': drug,
                    'target': target,
                    'binding_score': binding_score
                })
        
        drug_targets_df = pd.DataFrame(drug_target_pairs)
        
        self.logger.info(f"Loaded {len(drug_targets_df)} drug-target associations")
        return drug_targets_df
    
    def _calculate_gene_risk_scores(self, model_dict: Dict[str, Any], 
                                   features: Dict[str, Any]) -> pd.Series:
        """Calculate risk scores for genes using trained model"""
        
        model = model_dict['model']
        X = features['integrated_features']
        
        # Get prediction probabilities (disease risk)
        risk_scores = model.predict_proba(X)[:, 1]
        
        # Create Series with gene names as index
        gene_risk_series = pd.Series(risk_scores, index=X.index)
        
        self.logger.info(f"Calculated risk scores for {len(gene_risk_series)} genes")
        return gene_risk_series
    
    def _calculate_drug_scores(self, drug_targets: pd.DataFrame, 
                             gene_risk_scores: pd.Series,
                             features: Dict[str, Any]) -> pd.DataFrame:
        """Calculate drug scores using weighted aggregation"""
        
        # Get network features for weighting
        network_features = features.get('network_features', pd.DataFrame())
        
        drug_scores = []
        
        for drug in drug_targets['drug'].unique():
            # Get targets for this drug
            drug_data = drug_targets[drug_targets['drug'] == drug]
            targets = drug_data['target'].tolist()
            binding_scores = drug_data['binding_score'].tolist()
            
            # Calculate composite score
            composite_score = self._calculate_composite_score(
                targets, binding_scores, gene_risk_scores, network_features
            )
            
            # Additional drug metrics
            drug_metrics = self._calculate_drug_metrics(
                targets, gene_risk_scores, network_features
            )
            
            drug_scores.append({
                'drug': drug,
                'composite_score': composite_score,
                'n_targets': len(targets),
                'max_target_risk': drug_metrics['max_risk'],
                'mean_target_risk': drug_metrics['mean_risk'],
                'weighted_risk': drug_metrics['weighted_risk'],
                'network_importance': drug_metrics['network_importance'],
                'targets': ','.join(targets)
            })
        
        return pd.DataFrame(drug_scores)
    
    def _calculate_composite_score(self, targets: List[str], 
                                 binding_scores: List[float],
                                 gene_risk_scores: pd.Series,
                                 network_features: pd.DataFrame) -> float:
        """Calculate composite drug score using weighted aggregation"""
        
        if not targets:
            return 0.0
        
        total_score = 0.0
        total_weight = 0.0
        
        for target, binding_score in zip(targets, binding_scores):
            # Get risk score for target
            risk_score = gene_risk_scores.get(target, 0.0)
            
            # Calculate weight based on multiple factors
            weight = self._calculate_target_weight(target, binding_score, network_features)
            
            # Weighted contribution
            total_score += risk_score * weight
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            composite_score = total_score / total_weight
        else:
            composite_score = 0.0
        
        return composite_score
    
    def _calculate_target_weight(self, target: str, binding_score: float,
                               network_features: pd.DataFrame) -> float:
        """Calculate weight for a target gene"""
        
        # Base weight from binding affinity
        weight = binding_score
        
        # Add network-based weights if available
        if not network_features.empty and target in network_features.index:
            # Weight by network centrality
            if 'degree_centrality' in network_features.columns:
                centrality = network_features.loc[target, 'degree_centrality']
                weight *= (1 + centrality)
            
            # Weight by pathway importance
            if 'n_pathways' in network_features.columns:
                pathway_count = network_features.loc[target, 'n_pathways']
                weight *= (1 + np.log1p(pathway_count))
        
        return weight
    
    def _calculate_drug_metrics(self, targets: List[str],
                              gene_risk_scores: pd.Series,
                              network_features: pd.DataFrame) -> Dict[str, float]:
        """Calculate additional drug metrics"""
        
        # Get risk scores for targets
        target_risks = [gene_risk_scores.get(target, 0.0) for target in targets]
        
        if not target_risks:
            return {
                'max_risk': 0.0,
                'mean_risk': 0.0,
                'weighted_risk': 0.0,
                'network_importance': 0.0
            }
        
        # Calculate network importance
        network_importance = 0.0
        if not network_features.empty:
            target_centralities = []
            for target in targets:
                if target in network_features.index and 'degree_centrality' in network_features.columns:
                    centrality = network_features.loc[target, 'degree_centrality']
                    target_centralities.append(centrality)
            
            if target_centralities:
                network_importance = np.mean(target_centralities)
        
        metrics = {
            'max_risk': max(target_risks),
            'mean_risk': np.mean(target_risks),
            'weighted_risk': np.mean(target_risks),  # Could be more sophisticated
            'network_importance': network_importance
        }
        
        return metrics
    
    def get_drug_target_details(self, drug: str, 
                              drug_targets: pd.DataFrame,
                              gene_risk_scores: pd.Series) -> Dict[str, Any]:
        """Get detailed information about a drug's targets"""
        
        drug_data = drug_targets[drug_targets['drug'] == drug]
        
        target_details = []
        for _, row in drug_data.iterrows():
            target = row['target']
            binding_score = row['binding_score']
            risk_score = gene_risk_scores.get(target, 0.0)
            
            target_details.append({
                'target': target,
                'binding_score': binding_score,
                'risk_score': risk_score,
                'contribution': binding_score * risk_score
            })
        
        # Sort by contribution
        target_details.sort(key=lambda x: x['contribution'], reverse=True)
        
        return {
            'drug': drug,
            'n_targets': len(target_details),
            'targets': target_details
        }
    
    def save_results(self, drug_scores: pd.DataFrame, filepath: Path):
        """Save drug scoring results"""
        drug_scores.to_csv(filepath, index=False)
        self.logger.info(f"Saved drug scores to {filepath}")
    
    def load_results(self, filepath: Path) -> pd.DataFrame:
        """Load drug scoring results"""
        drug_scores = pd.read_csv(filepath)
        self.logger.info(f"Loaded drug scores from {filepath}")
        return drug_scores
    
    def get_top_drugs(self, drug_scores: pd.DataFrame, n_top: int = 10) -> pd.DataFrame:
        """Get top N drugs by composite score"""
        return drug_scores.head(n_top)
    
    def filter_drugs_by_criteria(self, drug_scores: pd.DataFrame,
                               min_targets: int = 2,
                               min_score: float = 0.5) -> pd.DataFrame:
        """Filter drugs based on criteria"""
        
        filtered = drug_scores[
            (drug_scores['n_targets'] >= min_targets) &
            (drug_scores['composite_score'] >= min_score)
        ]
        
        self.logger.info(f"Filtered to {len(filtered)} drugs based on criteria")
        return filtered