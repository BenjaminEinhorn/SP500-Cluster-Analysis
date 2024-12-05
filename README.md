# S&P 500 Cluster Analysis

This project analyzes S&P 500 stock data by:
- Fetching stock data from Yahoo Finance.
- Calculating daily returns and correlation matrices.
- Identifying stock clusters using the Louvain community detection algorithm.
- Visualizing clusters and saving cluster information and statistics.

## Features
1. **Data Interpolation**: Handles missing stock data by interpolating time-series values.
2. **Correlation Analysis**: Computes the correlation matrix for S&P 500 stocks.
3. **Graph-Based Clustering**: Uses NetworkX to create a graph of stock correlations and applies the Louvain algorithm to detect clusters.
4. **Visualization**: Generates visual graphs for each cluster.
5. **Export Results**: Saves solitary tickers, clusters, and inter-cluster correlation statistics to text and CSV files.

## Requirements
Install the dependencies listed in the `requirements.txt` file:
```bash
pip install -r requirements.txt
