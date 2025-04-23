import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def safe_int(x):
    """
    ry to convert x to an integer; 
    if x is a numeric character or a numeric type, it will be converted normally. 
    Otherwise (including 'N/A', None, word, empty string), it will return 0.
    """
    try:
        #  float() befor
        return int(float(x))
    except:
        return 0

def safe_float(x):
    try:
        return float(x)
    except:
        return 0.0

# Loading dataset
print("Loading dataset...")
dataset = load_dataset("GEM/mlb_data_to_text", split="train")
df_games = dataset.to_pandas()
print(f"Loaded {len(df_games)} games.")


# 2. Flattening box_score
print("Flattening box_score into player-level records...")
records = []
for _, game in df_games.iterrows():
    for p in game['box_score']:
        ab   = safe_int(p.get('ab'))
        h    = safe_int(p.get('h'))
        r    = safe_int(p.get('r'))
        rbi  = safe_int(p.get('rbi'))
        hr   = safe_int(p.get('hr'))
        so   = safe_int(p.get('so'))
        bb   = safe_int(p.get('bb'))
        lob  = safe_int(p.get('lob'))
        avg  = safe_float(p.get('avg'))

        records.append({
            'player_id': p.get('player_id'),
            'team':      p.get('team'),
            'home_city': game['home_city'],
            'home_team': game['home_name'],
            'ab':        ab,
            'h':         h,
            'r':         r,
            'rbi':       rbi,
            'hr':        hr,
            'so':        so,
            'bb':        bb,
            'lob':       lob,
            'avg':       avg,
        })

df = pd.DataFrame(records)

# 3. hit_rate
print("Computing hit_rate...")
df = df[df['ab'] > 0].copy()
df['hit_rate'] = df['h'] / df['ab']

# 4. Feature Selection & Normalization
features = ['hit_rate', 'rbi', 'hr', 'so', 'bb', 'avg']
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. KMeans
print("Running KMeans clustering...")
kmeans = KMeans(n_clusters=4, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# 6. PCA 
print("Applying PCA for visualization...")
pca = PCA(n_components=2, random_state=42)
proj = pca.fit_transform(X_scaled)
df['pca1'], df['pca2'] = proj[:,0], proj[:,1]

# 7. Visualization
plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='pca1', y='pca2',
                hue='cluster', palette='tab10', alpha=0.6)
plt.title('PCA Projection of Player Hitting Profiles (4 Clusters)')
plt.xlabel('PCA 1'); plt.ylabel('PCA 2')
plt.legend(title='Cluster')
plt.tight_layout()
plt.savefig("pca_projection.png", dpi=300, bbox_inches="tight")  
plt.show()

# 8. City - Clustering Heatmap
print("Drawing city × cluster heatmap...")
city_cluster = df.groupby(['home_city','cluster']).size().unstack(fill_value=0)
city_cluster_prop = city_cluster.div(city_cluster.sum(axis=1), axis=0)
plt.figure(figsize=(12,8))
sns.heatmap(city_cluster_prop, cmap='YlGnBu', linewidths=0.5)
plt.title('Proportion of Hitting Style Clusters by Home City')
plt.xlabel('Cluster'); plt.ylabel('Home City')
plt.tight_layout()
plt.savefig("city_cluster_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()


# 9. Comparing home vs away
print("Comparing home vs away hit_rate by cluster...")

# 9.1 calculate the average hit_rate
# df['is_home']：True indicates that the player's record is for a home game.
df['is_home'] = (df['team'] == df['home_team'])

# Group by 'cluster' and 'is_home' to calculate the average hit rate.
home_away = (
    df.groupby(['cluster', 'is_home'])['hit_rate']
      .mean()
      .unstack()  
      .rename(columns={False: 'away_hit_rate', True: 'home_hit_rate'})
)

home_away['delta'] = home_away['home_hit_rate'] - home_away['away_hit_rate']

print(home_away)  

# 9.2 Home vs Away
print("Plotting home vs away grouped bar chart...")
fig, ax = plt.subplots(figsize=(8, 5))
x = home_away.index.astype(str)  # ['0','1','2','3']
width = 0.35

# Away
ax.bar(x, home_away['away_hit_rate'], width=width,
       label='Away Hit Rate', alpha=0.7, color='C0')
# Home
ax.bar([int(i) + width for i in range(len(x))],
       home_away['home_hit_rate'], width=width,
       label='Home Hit Rate', alpha=0.7, color='C1')

# Polish
ax.set_xticks([i + width/2 for i in range(len(x))])
ax.set_xticklabels(x)
ax.set_xlabel('Cluster (Hitting Style)')
ax.set_ylabel('Average Hit Rate')
ax.set_title('Home vs Away Hit Rate by Hitting Style Cluster')
ax.legend()

plt.tight_layout()
fig.savefig("home_away_by_cluster.jpg", dpi=300, bbox_inches="tight")
plt.show()


# 9.3  Delta 
print("Plotting delta bar chart...")
fig, ax = plt.subplots(figsize=(6, 4))

# 9.4 t-tests
print("Performing t-tests for each cluster and saving results...")

from scipy.stats import ttest_ind

# Prepare a list to store the statistics for each cluster.
ttest_results = []

for c in home_away.index:
    sub = df[df['cluster'] == c]
    home_rates = sub.loc[sub['is_home'], 'hit_rate']
    away_rates = sub.loc[~sub['is_home'], 'hit_rate']
    # Welch’s t-test
    t_stat, p_val = ttest_ind(home_rates, away_rates, equal_var=False)
    ttest_results.append({
        'cluster':       c,
        'home_mean':     home_rates.mean(),
        'away_mean':     away_rates.mean(),
        'delta':         home_rates.mean() - away_rates.mean(),
        't_statistic':   t_stat,
        'p_value':       p_val
    })

#  DataFrame
ttest_df = pd.DataFrame(ttest_results)
print(ttest_df.to_string(index=False))

# Save as CSV
ttest_df.to_csv("t_test_results_by_cluster.csv", index=False)
print("T-test results saved to t_test_results_by_cluster.csv")

# seaborn - color 
sns.barplot(
    x=home_away.reset_index()['cluster'].astype(str),
    y=home_away['delta'].values,
    color='C2',
    ax=ax
)

# 0 line
ax.axhline(0, color='gray', linewidth=1)
ax.set_xlabel('Cluster')
ax.set_ylabel('Home − Away Hit Rate')
ax.set_title('Home‑Away Hit Rate Difference by Style')

plt.tight_layout()
fig.savefig("delta_by_cluster.jpg", dpi=300, bbox_inches="tight")
plt.show()

print("Analysis complete.")
