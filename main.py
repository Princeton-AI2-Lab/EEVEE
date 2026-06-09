#!/usr/bin/env python3
"""EEVEE entry point: load a config file and run the pipeline."""
import os
import sys
import yaml
from pathlib import Path
from evolve.pipeline import EEVEEPipeline

def main():
    repo_root = Path(__file__).parent.resolve()
    config_path = repo_root / "configs" / "demo.yaml"
    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1]).resolve()
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not config:
        config = {}
    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root))
    pipeline = EEVEEPipeline(config)
    pipeline.run()

if __name__ == "__main__":
    main()
