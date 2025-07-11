# src/feature_engineer.py
"""
Feature Engineering Module
Handles dimensionality reduction, multi-view fusion, and feature integration
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Tuple
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

class FeatureEngineer:
    """Feature engineering for multi-omics integration"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.pca_models = {}
        self.cca_model = None
        
    def create_integrated_features(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Create integrated feature matrix from multi-omics data"""
        
        # Step 1: Apply PCA to each omics dataset
        reduced_features = self._apply_pca_to_datasets(datasets)
        
        # Step 2: Create gene-centric feature matrix
        gene_features = self._create_gene_features(reduced_features, datasets)
        
        # Step 3: Add network-based features
        network_features = self._create_network_features(datasets['annotations'])
        
        # Step 4: Integrate all features
        integrated_features = self._integrate_features(gene_features, network_features)
        
        # Step 5: Apply multi-view fusion
        fused_features = self._apply_multiview_fusion(integrated_features)
        
        feature_dict = {
            'integrated_features': fused_features,
            'gene_features': gene_features,
            'network_features': network_features,
            'feature_names': self._get_feature_names()
        }
        
        return feature_dict
    
    def _apply_pca_to_datasets(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Apply PCA to each omics dataset"""
        reduced_features = {}
        
        for data_type, data in datasets.items():
            if data_type == 'annotations':
                continue
                
            n_components = self.config.get('pca_components', {}).get(data_type, 100)
            n_components = min(n_components, min(data.shape) - 1)
            
            self.logger.info(f"Applying PCA to {data_type} data with {n_components} components")
            
            pca = PCA(n_components=n_components, random_state=42)
            
            # Transpose so samples are rows
            data_transposed = data.T
            pca_result = pca.fit_transform(data_transposed)
            
            # Store PCA model
            self.pca_models[data_type] = pca
            
            # Create DataFrame with PCA results
            pca_df = pd.DataFrame(
                pca_result.T,
                index=[f"{data_type}_PC{i+1}" for i in range(n_components)],
                columns=data.columns
            )
            
            reduced_features[data_type] = pca_df
            
            explained_variance = pca.explained_variance_ratio_.sum()
            self.logger.info(f"PCA for {data_type}: {explained_variance:.3f} variance explained")
        
        return reduced_features
    
    def _create_gene_features(self, reduced_features: Dict[str, pd.DataFrame], 
                            original_datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create gene-centric feature matrix"""
        
        # Get common genes across all datasets
        gene_sets = []
        for data_type, data in original_datasets.items():
            if data_type != 'annotations':
                # Convert feature IDs to gene IDs (simplified mapping)
                gene_ids = self._map_features_to_genes(data.index, data_type)
                gene_sets.append(set(gene_ids))
        
        common_genes = set.intersection(*gene_sets) if gene_sets else set()
        common_genes = list(common_genes)[:5000]  # Limit for demo
        
        self.logger.info(f"Found {len(common_genes)} common genes")
        
        # Create feature matrix for each gene
        gene_features_list = []
        
        for gene in common_genes:
            gene_feature_vector = []
            
            # Add features from each omics type
            for data_type, pca_data in reduced_features.items():
                # Get representative features for this gene
                gene_features = self._get_gene_features_from_pca(gene, pca_data, data_type)
                gene_feature_vector.extend(gene_features)
            
            gene_features_list.append(gene_feature_vector)
        
        # Create DataFrame
        feature_columns = self._get_feature_column_names(reduced_features)
        gene_features_df = pd.DataFrame(
            gene_features_list,
            index=common_genes,
            columns=feature_columns
        )
        
        return gene_features_df
    
    def _map_features_to_genes(self, feature_ids: List[str], data_type: str) -> List[str]:
        """Map feature IDs to gene IDs"""
        # Simplified mapping - in practice, use proper ID mapping
        if data_type == 'transcriptomics':
            return [f"GENE_{i}" for i in range(len(feature_ids))]
        elif data_type == 'proteomics':
            return [f"GENE_{i}" for i in range(len(feature_ids))]
        elif data_type == 'epigenomics':
            # Map genomic regions to nearby genes
            return [f"GENE_{i % 10000}" for i in range(len(feature_ids))]
        elif data_type == 'metabolomics':
            # Map metabolites to associated genes
            return [f"GENE_{i % 8000}" for i in range(len(feature_ids))]
        else:
            return feature_ids
    
    def _get_gene_features_from_pca(self, gene: str, pca_data: pd.DataFrame, 
                                   data_type: str) -> List[float]:
        """Extract features for a gene from PCA data"""
        # For demo purposes, return mean of random components
        # In practice, would use proper gene-to-feature mapping
        np.random.seed(hash(gene + data_type) % 2147483647)
        n_components = len(pca_data)
        selected_components = np.random.choice(range(n_components), 
                                             size=min(10, n_components), 
                                             replace=False)
        
        features = []
        for comp in selected_components:
            # Use mean across samples as representative feature
            feature_value = pca_data.iloc[comp].mean()
            features.append(feature_value)
        
        return features
    
    def _get_feature_column_names(self, reduced_features: Dict[str, pd.DataFrame]) -> List[str]:
        """Generate feature column names"""
        columns = []
        for data_type, pca_data in reduced_features.items():
            n_features = min(10, len(pca_data))
            for i in range(n_features):
                columns.append(f"{data_type}_feature_{i+1}")
        return columns
    
    def _create_network_features(self, annotations: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create network-based features"""
        
        # Create protein-protein interaction network
        ppi_network = self._create_ppi_network(annotations['ppi'])
        
        # Calculate network metrics
        network_metrics = self._calculate_network_metrics(ppi_network)
        
        # Create pathway features
        pathway_features = self._create_pathway_features(annotations['pathways'])
        
        # Create GO features
        go_features = self._create_go_features(annotations['go'])
        
        # Combine all network features
        network_features = pd.concat([
            network_metrics,
            pathway_features,
            go_features
        ], axis=1, sort=False)
        
        return network_features
    
    def _create_ppi_network(self, ppi_data: pd.DataFrame) -> nx.Graph:
        """Create protein-protein interaction network"""
        G = nx.Graph()
        
        for _, row in ppi_data.iterrows():
            G.add_edge(row['protein1'], row['protein2'], weight=row['score'])
        
        self.logger.info(f"Created PPI network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G
    
    def _calculate_network_metrics(self, network: nx.Graph) -> pd.DataFrame:
        """Calculate network-based metrics for genes"""
        metrics = {}
        
        # Degree centrality
        degree_centrality = nx.degree_centrality(network)
        
        # Betweenness centrality
        betweenness_centrality = nx.betweenness_centrality(network, k=1000)  # Sample for speed
        
        # Closeness centrality
        closeness_centrality = nx.closeness_centrality(network)
        
        # PageRank
        pagerank = nx.pagerank(network)
        
        # Clustering coefficient
        clustering = nx.clustering(network)
        
        # Combine metrics
        all_nodes = set(network.nodes())
        
        metrics_data = []
        for node in all_nodes:
            metrics_data.append({
                'degree_centrality': degree_centrality.get(node, 0),
                'betweenness_centrality': betweenness_centrality.get(node, 0),
                'closeness_centrality': closeness_centrality.get(node, 0),
                'pagerank': pagerank.get(node, 0),
                'clustering': clustering.get(node, 0)
            })
        
        metrics_df = pd.DataFrame(metrics_data, index=list(all_nodes))
        
        self.logger.info(f"Calculated network metrics for {len(metrics_df)} nodes")
        return metrics_df
    
    def _create_pathway_features(self, pathway_data: pd.DataFrame) -> pd.DataFrame:
        """Create pathway-based features"""
        
        # Create binary matrix of gene-pathway associations
        genes = pathway_data['gene'].unique()
        pathways = pathway_data['pathway'].unique()
        
        pathway_matrix = pd.DataFrame(0, index=genes, columns=pathways)
        
        for _, row in pathway_data.iterrows():
            pathway_matrix.loc[row['gene'], row['pathway']] = 1
        
        # Calculate pathway statistics
        pathway_features = pd.DataFrame(index=genes)
        
        # Number of pathways per gene
        pathway_features['n_pathways'] = pathway_matrix.sum(axis=1)
        
        # Pathway diversity (entropy)
        pathway_features['pathway_entropy'] = pathway_matrix.apply(
            lambda x: -sum(p * np.log2(p + 1e-10) for p in x if p > 0), axis=1
        )
        
        self.logger.info(f"Created pathway features for {len(pathway_features)} genes")
        return pathway_features
    
    def _create_go_features(self, go_data: pd.DataFrame) -> pd.DataFrame:
        """Create Gene Ontology features"""
        
        genes = go_data['gene'].unique()
        go_terms = go_data['go_term'].unique()
        
        # Create GO term categories (simplified)
        bp_terms = [t for t in go_terms if hash(t) % 3 == 0]  # Biological Process
        mf_terms = [t for t in go_terms if hash(t) % 3 == 1]  # Molecular Function
        cc_terms = [t for t in go_terms if hash(t) % 3 == 2]  # Cellular Component
        
        go_features = pd.DataFrame(index=genes)
        
        # Count GO terms by category
        for gene in genes:
            gene_terms = set(go_data[go_data['gene'] == gene]['go_term'])
            
            go_features.loc[gene, 'bp_terms'] = len(gene_terms.intersection(bp_terms))
            go_features.loc[gene, 'mf_terms'] = len(gene_terms.intersection(mf_terms))
            go_features.loc[gene, 'cc_terms'] = len(gene_terms.intersection(cc_terms))
            go_features.loc[gene, 'total_go_terms'] = len(gene_terms)
        
        go_features = go_features.fillna(0)
        
        self.logger.info(f"Created GO features for {len(go_features)} genes")
        return go_features
    
    def _integrate_features(self, gene_features: pd.DataFrame, 
                          network_features: pd.DataFrame) -> pd.DataFrame:
        """Integrate gene features with network features"""
        
        # Find common genes
        common_genes = gene_features.index.intersection(network_features.index)
        
        # Align dataframes
        gene_features_aligned = gene_features.loc[common_genes]
        network_features_aligned = network_features.loc[common_genes]
        
        # Concatenate features
        integrated = pd.concat([gene_features_aligned, network_features_aligned], axis=1)
        
        # Handle missing values
        integrated = integrated.fillna(0)
        
        self.logger.info(f"Integrated features for {len(integrated)} genes with {len(integrated.columns)} features")
        return integrated
    
    def _apply_multiview_fusion(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply multi-view data fusion using CCA"""
        
        # Split features into views based on data type
        views = self._split_features_into_views(features)
        
        if len(views) < 2:
            self.logger.warning("Not enough views for CCA, returning original features")
            return features
        
        # Apply CCA between first two views
        view1_cols = views[0]
        view2_cols = views[1]
        
        view1_data = features[view1_cols].values
        view2_data = features[view2_cols].values
        
        # Determine number of components
        n_components = min(len(view1_cols), len(view2_cols), 50)
        
        cca = CCA(n_components=n_components)
        view1_cca, view2_cca = cca.fit_transform(view1_data, view2_data)
        
        self.cca_model = cca
        
        # Combine CCA results
        cca_features = np.hstack([view1_cca, view2_cca])
        
        # Add remaining views
        remaining_views = views[2:] if len(views) > 2 else []
        if remaining_views:
            remaining_data = features[sum(remaining_views, [])].values
            cca_features = np.hstack([cca_features, remaining_data])
        
        # Create DataFrame
        feature_names = ([f"CCA1_{i}" for i in range(n_components)] +
                        [f"CCA2_{i}" for i in range(n_components)] +
                        sum(remaining_views, []))
        
        fused_features = pd.DataFrame(
            cca_features,
            index=features.index,
            columns=feature_names
        )
        
        self.logger.info(f"Applied CCA fusion: {fused_features.shape}")
        return fused_features
    
    def _split_features_into_views(self, features: pd.DataFrame) -> List[List[str]]:
        """Split features into different views based on data type"""
        views = []
        
        # Group features by data type
        transcriptomics_features = [col for col in features.columns if 'transcriptomics' in col]
        proteomics_features = [col for col in features.columns if 'proteomics' in col]
        epigenomics_features = [col for col in features.columns if 'epigenomics' in col]
        metabolomics_features = [col for col in features.columns if 'metabolomics' in col]
        network_features = [col for col in features.columns if any(net in col for net in ['centrality', 'pathway', 'go', 'clustering'])]
        
        if transcriptomics_features:
            views.append(transcriptomics_features)
        if proteomics_features:
            views.append(proteomics_features)
        if epigenomics_features:
            views.append(epigenomics_features)
        if metabolomics_features:
            views.append(metabolomics_features)
        if network_features:
            views.append(network_features)
        
        return views
    
    def _get_feature_names(self) -> List[str]:
        """Get feature names for the final feature matrix"""
        # This would be populated during feature creation
        return []
    
    def save_features(self, features: Dict[str, Any], filepath: Path):
        """Save features to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(features, f)
        self.logger.info(f"Saved features to {filepath}")
    
    def load_features(self, filepath: Path) -> Dict[str, Any]:
        """Load features from file"""
        with open(filepath, 'rb') as f:
            features = pickle.load(f)
        self.logger.info(f"Loaded features from {filepath}")
        return features

class ModelTrainer:
    """Model trainer for multi-omics integration pipeline"""
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def train(self, features: dict) -> dict:
        """Train a model on the integrated features. Returns a dict with the model and metadata."""
        X = features['integrated_features']
        # For demonstration, generate synthetic labels
        y = (X.sum(axis=1) > X.sum(axis=1).median()).astype(int)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        self.logger.info(f"Trained RandomForest model on {X.shape[0]} samples, {X.shape[1]} features.")
        return {'model': model}

    def save_model(self, model_dict: dict, filepath: Path):
        """Save the trained model to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump(model_dict, f)
        self.logger.info(f"Saved model to {filepath}")

    def load_model(self, filepath: Path) -> dict:
        """Load a trained model from a file."""
        with open(filepath, 'rb') as f:
            model_dict = pickle.load(f)
        self.logger.info(f"Loaded model from {filepath}")
        return model_dict