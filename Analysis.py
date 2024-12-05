import yfinance as yf
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from community import community_louvain
import matplotlib.colors as mcolors
import os

# Set working directory
os.chdir(r'E:\Data\S&P Analysis')

# Step 1: Fetch data
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
sp500_table = pd.read_html(url)[0]
tickers = sp500_table['Symbol'].tolist()

# Uncomment the following lines if data is to be fetched from Yahoo Finance
data = yf.download(tickers, start="2015-01-01", end="2023-12-31")['Adj Close']


# Interpolate missing values based on time trends
data = data.interpolate(method='time', limit_direction='both')  # Interpolating NaN values with respect to time
print(f"Number of valid tickers after interpolation: {data.shape[1]}")

# Step 2: Calculate returns and correlations
returns = data.pct_change(fill_method=None)
correlation_matrix = returns.corr()

# Step 3: Build adjacency matrix with a higher threshold
threshold = 0.7  # Set a higher threshold for stronger correlations
adjacency_matrix = (correlation_matrix > threshold).astype(int)

# Step 4: Create graph
G = nx.from_numpy_array(adjacency_matrix.to_numpy())
mapping = {i: ticker for i, ticker in enumerate(correlation_matrix.columns)}
G = nx.relabel_nodes(G, mapping)

# Add edge weights (correlation values)
for i, j in G.edges():
    G[i][j]['weight'] = float(correlation_matrix.loc[i, j])

# Step 5: Identify clusters using Louvain
partition = community_louvain.best_partition(G)
nx.set_node_attributes(G, partition, 'cluster')

# Step 6: Extract clusters
clusters = {}
for node, cluster_id in partition.items():
    clusters.setdefault(cluster_id, []).append(node)

# Separate solitary tickers and clusters with multiple tickers
solitary_tickers = [tickers[0] for tickers in clusters.values() if len(tickers) == 1]
multi_ticker_clusters = {k: v for k, v in clusters.items() if len(v) > 1}

# Output directories
output_dir = r"E:\Data\S&P Analysis\Clusters"
solitary_dir = os.path.join(output_dir, "Solitary Tickers")
multi_cluster_dir = os.path.join(output_dir, "Cluster_Tickets")
os.makedirs(solitary_dir, exist_ok=True)
os.makedirs(multi_cluster_dir, exist_ok=True)

# Step 7: Save solitary tickers
solitary_output_path = os.path.join(solitary_dir, "Solitary_Tickers.txt")
with open(solitary_output_path, 'w') as f:
    for ticker in solitary_tickers:
        f.write(f"{ticker}\n")
print(f"Solitary tickers saved to {solitary_output_path}")

# Save multi-ticker clusters
for cluster_id, nodes in multi_ticker_clusters.items():
    cluster_file_path = os.path.join(multi_cluster_dir, f"Cluster_{cluster_id}_Tickers.txt")
    with open(cluster_file_path, 'w') as f:
        for ticker in nodes:
            f.write(f"{ticker}\n")
    print(f"Tickers for Cluster {cluster_id} saved to {cluster_file_path}")

# Step 8: Calculate inter-cluster correlations
cluster_correlation_info = {cluster_id: {"positive": [], "negative": [], "neutral": []} for cluster_id in multi_ticker_clusters.keys()}

for cluster_id_1, nodes_1 in multi_ticker_clusters.items():
    for cluster_id_2, nodes_2 in multi_ticker_clusters.items():
        if cluster_id_1 == cluster_id_2:
            continue

        # Filter the correlation matrix for nodes in both clusters
        cluster_corr = correlation_matrix.loc[nodes_1, nodes_2]
        inter_cluster_corr = cluster_corr.values.mean()

        # Categorize the correlation
        if inter_cluster_corr > threshold:
            cluster_correlation_info[cluster_id_1]["positive"].append(cluster_id_2)
        elif inter_cluster_corr < -threshold:
            cluster_correlation_info[cluster_id_1]["negative"].append(cluster_id_2)
        else:
            cluster_correlation_info[cluster_id_1]["neutral"].append(cluster_id_2)

# Step 9: Save cluster statistics and generate graphs
stats_dir = os.path.join(output_dir, "Cluster Statistics")
os.makedirs(stats_dir, exist_ok=True)

cluster_stats = []
colors = list(mcolors.TABLEAU_COLORS.values())
cluster_colors = {cluster: colors[i % len(colors)] for i, cluster in enumerate(multi_ticker_clusters.keys())}

for cluster_id, nodes in multi_ticker_clusters.items():
    graph_output_path = os.path.join(output_dir, f"Cluster_{cluster_id}_Size_{len(nodes)}.jpeg")

    try:
        # Generate graph for clusters with more than one ticker
        subgraph = G.subgraph(nodes)
        pos = nx.spring_layout(subgraph, seed=42)

        # Plot the subgraph
        plt.figure(figsize=(10, 10))
        nx.draw_networkx_nodes(subgraph, pos, node_size=500, node_color=cluster_colors[cluster_id], alpha=0.9)
        nx.draw_networkx_edges(subgraph, pos, alpha=0.5)
        nx.draw_networkx_labels(subgraph, pos, font_size=8, font_color="black",
                                bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"))
        plt.title(f"Cluster {cluster_id} (Size: {len(nodes)})", fontsize=16)
        plt.axis("off")

        # Save the figure
        plt.savefig(graph_output_path, format="jpeg", dpi=300)
        plt.close()
        print(f"Cluster {cluster_id} graph saved to {graph_output_path}")
    except Exception as e:
        print(f"Failed to save graph for Cluster {cluster_id}: {e}")

    # Save statistics for the cluster
    cluster_stats.append({
        "Cluster ID": cluster_id,
        "Size": len(nodes),
        "Tickers": ", ".join(nodes),
        "Positively Correlated Clusters": ", ".join(map(str, cluster_correlation_info[cluster_id]["positive"])),
        "Negatively Correlated Clusters": ", ".join(map(str, cluster_correlation_info[cluster_id]["negative"])),
        "Neutral Clusters": ", ".join(map(str, cluster_correlation_info[cluster_id]["neutral"])),
        "Graph Path": graph_output_path,
    })

# Save cluster statistics to a CSV file
cluster_stats_df = pd.DataFrame(cluster_stats)
stats_output_path = os.path.join(stats_dir, "Cluster_Statistics.csv")
cluster_stats_df.to_csv(stats_output_path, index=False)

print(f"Cluster statistics saved to {stats_output_path}")
