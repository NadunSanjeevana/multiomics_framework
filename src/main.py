# main.py
"""
Multi-Omics Integration Pipeline for Disease Gene Prioritization and Drug Repurposing
Main execution script
"""

import argparse
import logging
import os
from pathlib import Path
import yaml
import sys
sys.path.append(str(Path(__file__).resolve().parent))

from data_loader import MultiOmicsDataLoader
from preprocessor import MultiOmicsPreprocessor
from feature_engineer import FeatureEngineer
from model_trainer import ModelTrainer
from drug_scorer import DrugScorer
from utils import setup_logging, load_config

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Multi-Omics Integration Pipeline')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['train', 'predict', 'full'],
                       default='full', help='Execution mode')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting Multi-Omics Integration Pipeline")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Initialize pipeline components
        data_loader = MultiOmicsDataLoader(config['data'])
        preprocessor = MultiOmicsPreprocessor(config['preprocessing'])
        feature_engineer = FeatureEngineer(config['feature_engineering'])
        model_trainer = ModelTrainer(config['model'])
        drug_scorer = DrugScorer(config['drug_scoring'])

        # 1. Multi-omics Data Acquisition & Integration
        logger.info("Loading and integrating multi-omics datasets (transcriptomics, proteomics, epigenomics, metabolomics)...")
        raw_data = data_loader.load_all_datasets()
        # raw_data should be a dict: { 'transcriptomics': ..., 'proteomics': ..., 'epigenomics': ..., 'metabolomics': ... }

        # 2. Preprocessing & Normalization
        logger.info("Preprocessing and normalizing datasets (TPM, z-score, etc.) and mapping to HGNC gene IDs...")
        processed_data = preprocessor.preprocess_all(raw_data)
        # processed_data: normalized, mapped to unified gene IDs

        # 3. Feature Engineering: Concatenation & Enrichment
        logger.info("Engineering integrated features: concatenating omics layers and enriching with pathway, PPI, and annotation data...")
        features = feature_engineer.create_integrated_features(processed_data)
        # features: includes pathway, PPI, and annotation enrichment

        # 4. Dimensionality Reduction (PCA per omics layer)
        logger.info("Applying dimensionality reduction (PCA) to each omics layer...")
        # Already handled in feature_engineer.create_integrated_features()

        # 5. Multi-view Data Fusion (CCA/MKL)
        logger.info("Fusing multi-omics features using multi-view data fusion (CCA/MKL)...")
        # Already handled in feature_engineer.create_integrated_features()
        fused_features = features

        if args.mode in ['train', 'full']:
            # 6. Model Training
            logger.info("Training model on fused features...")
            model = model_trainer.train(fused_features)
            # Save model and features
            model_trainer.save_model(model, output_dir / 'model.pkl')
            feature_engineer.save_features(fused_features, output_dir / 'features.pkl')

        if args.mode in ['predict', 'full']:
            if args.mode == 'predict':
                logger.info("Loading trained model and features...")
                model = model_trainer.load_model(output_dir / 'model.pkl')
                fused_features = feature_engineer.load_features(output_dir / 'features.pkl')

            # 7. Weighted Drug Scoring
            logger.info("Scoring drugs using weighted aggregate of gene risk scores (network centrality, pathway importance, etc.)...")
            # Use score_drugs (already weighted) for now
            drug_scores = drug_scorer.score_drugs(model, fused_features)
            drug_scorer.save_results(drug_scores, output_dir / 'drug_scores.csv')

        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main()