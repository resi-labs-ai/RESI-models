import asyncio
import json
import logging
from pathlib import Path
import numpy as np
import yaml

from real_estate.data import parse_feature_config, TabularEncoder, create_default_feature_config
from real_estate.orchestration import ValidationOrchestrator
from real_estate.data.config_encoder import LEGACY_FEATURES

# Configure logging to see the new logs
logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(name)s | %(message)s')
logger = logging.getLogger(__name__)

async def test_validator_workflow():
    print("\n=== Starting Validator Workflow Simulation (Legacy Model Support) ===")
    
    # 1. Load Real Features from YAML
    yaml_path = Path("real_estate/data/mappings/feature_config.yaml")
    with open(yaml_path) as f:
        yaml_data = yaml.safe_load(f)
    
    real_features = list(yaml_data["feature_order"])
    
    # 2. Mock Feature Configs
    # Legacy model (79 features: 76 real + 3 legacy)
    legacy_features = real_features + list(LEGACY_FEATURES)
    legacy_config_raw = {
        "version": "1.0",
        "legacy_model": True,
        "features": legacy_features
    }
    
    # Modern model (76 features)
    modern_config_raw = {
        "version": "1.0",
        "features": real_features
    }
    
    # 3. Mock Metagraph and Model Paths
    hotkeys = ["hk_legacy", "hk_modern"]
    model_paths = {hk: Path(f"/tmp/{hk}.onnx") for hk in hotkeys}
    feature_configs = {
        "hk_legacy": parse_feature_config(legacy_config_raw),
        "hk_modern": parse_feature_config(modern_config_raw)
    }
    
    # 4. Dummy Property Data
    # Fill with 1.0 for all features (including legacy) to test zeroing
    properties = [{name: 1.0 for name in legacy_features}]
    
    # 5. Run Encoding (Simulation of ValidationOrchestrator._encode_models)
    print("\n[STEP] Running Feature Encoding...")
    encoded_models = ValidationOrchestrator._encode_models(
        model_paths=model_paths,
        feature_configs=feature_configs,
        properties=properties
    )
    
    # 6. Verification
    print("\n[STEP] Verifying Results...")
    for hk in hotkeys:
        features = encoded_models.features[hk]
        layout = encoded_models.layouts[hk]
        print(f"Hotkey: {hk}, Shape: {features.shape}, Legacy: {feature_configs[hk].legacy_model}")
        
        if feature_configs[hk].legacy_model:
            # Verify zero-filling
            for name in LEGACY_FEATURES:
                idx = layout.feature_names.index(name)
                val = features[0, idx]
                print(f"  Legacy feature '{name}' at index {idx} value: {val}")
                assert val == 0.0, f"Legacy feature {name} was not zeroed!"
    
    print("\n=== Simulation Complete: Legacy models are correctly identified and zero-filled ===")

if __name__ == "__main__":
    asyncio.run(test_validator_workflow())
