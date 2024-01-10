# Music Genre Clustering

This repository contains a Python project for unsupervised clustering of songs based on their musical features.

## Overview

The project uses a dataset of songs with various attributes such as popularity, acousticness, danceability, energy, instrumentalness, key, liveness, loudness, mode, speechiness, tempo, and valence. It employs Principal Component Analysis (PCA) for dimensionality reduction and K-Means clustering for grouping similar songs.

## Project Structure

- `main.py`: The main script that orchestrates the reading of the dataset, the application of one-hot encoding to categorical variables, PCA transformation, K-Means clustering, and visualization using Plotly.
- `pca.py`: A module that provides functionality for applying PCA to the dataset.
- `kmeans.py`: A module that applies K-Means clustering to the PCA-transformed data.
- `app/data/sampled_dataset.csv`: The dataset containing the song attributes.
- `app/output/clustered_data.csv`: The output file with the PCA components and cluster labels.
- `app/output/cluster_plot.html`: An HTML file containing the interactive Plotly scatter plot of the clustered songs.

## Features

- **Data Preprocessing**: Categorical features like musical key and mode are one-hot encoded for use in the clustering algorithm.
- **Dimensionality Reduction**: PCA is applied to reduce the feature space while maintaining the variance in the dataset.
- **Clustering**: K-Means clustering groups songs into clusters based on their PCA-transformed features.
- **Visualization**: An interactive scatter plot is generated using Plotly, showing the clusters and allowing users to hover over points to see song details.

## Contributions
Contributions are welcome! Please feel free to submit a pull request or create an issue for any bugs or enhancements.

## License
This project is open-source and available under the MIT License.

## Acknowledgments
This project is inspired by the ability of machine learning to uncover patterns in complex datasets, such as those found in music attributes.
