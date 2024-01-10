import pandas as pd
import pca
import kmeans
import plotly.express as px

def main():   

    df = pd.read_csv('app/data/music_genre_unsupervised.csv')    

    # The first column 'track_name' will be used for hover information
    song_names = df['track_name']
    features_df = df[df.columns[1:]]  # This excludes the first column which is 'track_name'
    
    # Apply PCA from the pca module
    principalDf, pca_model = pca.apply_pca(features_df)
    
    # Apply K-Means from the kmeans module
    kmeans_model, labels = kmeans.apply_kmeans(principalDf)
    principalDf['cluster'] = labels  # Append cluster labels to the DataFrame
    
    # Append the song names to the principalDf
    principalDf['song_name'] = song_names

    # Save the DataFrame with principal components and cluster labels to a new CSV file
    principalDf.to_csv('app/output/clustered_data.csv', index=False)

     # Create a Plotly scatter plot of the clustered data with song names on hover
    fig = px.scatter(principalDf, x='principal component 1', y='principal component 2',color='cluster', hover_name='song_name',labels={'cluster': 'Cluster'},title='PCA Clustering with K-Means')
    fig.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=0.5, color='DarkSlateGrey')),selector=dict(mode='markers'))
    fig.show()

    # Save the figure as an HTML file
    fig.write_html('app/output/cluster_plot.html')

if __name__ == "__main__":
    main()