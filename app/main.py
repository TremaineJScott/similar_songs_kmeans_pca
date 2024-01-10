import pandas as pd
import pca
import kmeans
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder

def main():   
    df = pd.read_csv('app/data/sampled_dataset.csv')    
    
    # The first two columns will be used for hover information
    artist_names = df['artist_name']
    track_names  = df['track_name']
    
       # Separate the features for one-hot encoding and the numeric features
    categorical_features = df[['key', 'mode']]
    numeric_features = df.drop(columns=['artist_name', 'track_name', 'key', 'mode'])

    # Apply one-hot encoding to the categorical features
    encoder = OneHotEncoder(sparse=False)
    encoded_categorical = encoder.fit_transform(categorical_features)
    
    # Create a DataFrame with the encoded categorical features
    encoded_df = pd.DataFrame(encoded_categorical,columns=encoder.get_feature_names_out(['key', 'mode']))
    
    # Combine the numeric features with the encoded categorical features
    features_df = pd.concat([numeric_features.reset_index(drop=True),encoded_df.reset_index(drop=True)], axis=1)

    # Apply PCA from the pca module
    principalDf, pca_model = pca.apply_pca(features_df)
    
    # Apply K-Means from the kmeans module
    kmeans_model, labels = kmeans.apply_kmeans(principalDf)
    principalDf['cluster'] = labels  # Append cluster labels to the DataFrame
    
    # Append the artist and track names to the principalDf for hover information
    principalDf['artist_name'] = artist_names
    principalDf['track_name'] = track_names

    # Save the DataFrame with principal components and cluster labels to a new CSV file
    principalDf.to_csv('app/output/clustered_data.csv', index=False)

    # Create a Plotly scatter plot of the clustered data with artist and track names on hover
    fig = px.scatter(principalDf, x='principal component 1', y='principal component 2',color='cluster', hover_data=['artist_name', 'track_name'],labels={'cluster': 'Cluster'},title='PCA Clustering with K-Means')
    fig.update_traces(marker=dict(size=10, opacity=0.8,line=dict(width=0.5, color='DarkSlateGrey')),selector=dict(mode='markers'))
    fig.show()

    # Save the figure as an HTML file
    fig.write_html('app/output/cluster_plot.html')

if __name__ == "__main__":
    main()