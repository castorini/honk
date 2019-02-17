from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from .similarity_metric import SimilarityMetric

class CosineSimilarity(SimilarityMetric):
    def compute_similarity(self, data, target=None):
        if not target:
            target = self.target

        data = np.array(data)
        target = np.array(target)
        assert data.shape == target.shape
        
        similarity = cosine_similarity(np.expand_dims(data, axis=0), np.expand_dims(target, axis=0))
        return similarity[0][0]
