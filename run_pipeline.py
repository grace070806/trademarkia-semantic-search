"""
run_pipeline.py  —  Master pipeline runner.
Runs Part 1 (embeddings + ChromaDB) and Part 2 (Fuzzy Clustering),
then runs the threshold experiment.

Usage:
    python run_pipeline.py
"""
import logging, sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).parent))

from core.embeddings import run_part1
from core.clustering import run_part2

def main():
    logger.info("══ Part 1: Embeddings & Vector DB ══")
    run_part1(data_dir="./data")

    logger.info("══ Part 2: Fuzzy Clustering ══")
    run_part2(data_dir="./data", output_dir="./data/analysis")

    logger.info("══ Threshold Experiment ══")
    try:
        from sentence_transformers import SentenceTransformer
        from analysis.threshold_experiment import run_threshold_experiment
        model = SentenceTransformer("all-MiniLM-L6-v2")
        run_threshold_experiment(model, out_dir=Path("./data/analysis"))
    except Exception as e:
        logger.warning(f"Threshold experiment skipped: {e}")

    logger.info("\n✅ Pipeline complete.")
    logger.info("  uvicorn api.main:app --host 0.0.0.0 --port 8000")
    logger.info("  pytest tests/ -v")

if __name__ == "__main__":
    main()
