from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=10)

    def extract_features(self, data):
        scaled_data = self.scaler.fit_transform(data)
        features = self.pca.fit_transform(scaled_data)
        return features

