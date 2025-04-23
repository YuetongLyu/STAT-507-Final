# STAT-507-Final
# Exploring MLB Hitting Styles and Home-Field Effects

## Abstract  
This project analyzes Major League Baseball hitter performance across different home cities using the Hugging Face “GEM/mlb_data_to_text” dataset. We flatten per-game `box_score` data to player–game records, derive a per-game hit rate and other batting metrics, and cluster hitters into four archetypes (Contact, All-Rounder, Power, Patient) via K-Means. PCA validates cluster separability, a city–style heatmap reveals geographic flavor, and home vs. away comparisons (with Welch’s t-tests) quantify style-specific home-field advantages.

## Installation  
```bash
git clone https://github.com/your-username/mlb-hitting-styles.git
cd mlb-hitting-styles
python -m venv venv
source venv/bin/activate    # macOS/Linux
venv\Scripts\activate       # Windows
pip install -r requirements.txt
Data Preparation
No manual download is required. The script will fetch and cache the “GEM/mlb_data_to_text” dataset:

Files
mlb_analysis.py – all preprocessing, clustering, PCA, heatmaps, home/away analysis, t-tests and plotting

requirements.txt – Python dependencies

README.md – this file

LICENSE – MIT license

Results
After running, check results/ for:

pca_projection.png

city_cluster_heatmap.png

home_away_by_cluster.jpg

delta_by_cluster.jpg

t_test_results_by_cluster.csv

Citation
Lyu, Y. (2025). Exploring MLB Hitting Styles and Home-Field Effects. GitHub repository.
