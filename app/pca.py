from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

def apply_pca(data, n_components=2):

    # Standardize the features
    features = data.columns
    x = data.loc[:, features].values
    x = StandardScaler().fit_transform(x)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(x)
    
    # Create a DataFrame with the principal components
    principalDf = pd.DataFrame(data=principalComponents,columns=[f'principal component {i+1}' for i in range(n_components)])
    
    return principalDf, pca
