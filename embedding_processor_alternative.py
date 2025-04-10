import numpy as np
from database import get_db_connection, insert_embeddings

# THIS IS A DUMMY EMBEDDING FUNCTION FOR TESTING PURPOSES
def generate_embedding(text):
    """Generate a simple embedding for testing purposes."""
    try:
        # This is NOT for production use, just to test the database workflow
        np.random.seed(sum(ord(c) for c in text))
        # 384 dimensions to match the expected size
        return np.random.rand(384).tolist()  
    except Exception as e:
        print(f"Error generating embedding: {e}")
        # Return a dummy embedding
        return np.random.rand(384).tolist()
