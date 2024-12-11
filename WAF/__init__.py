from urllib.parse import unquote
import joblib
import numpy as np
import pickle
from abc import ABC, abstractmethod
from TrainingModels.model import TextVectorizer, SvmClassifier


class WAF(ABC):
    def __init__(self, model_path, vectorizer_path):
        # Load the saved model and vectorizer
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path

        self.model = SvmClassifier(mode="test")
        self.model.load(self.model_path)

        self.vectorizer = TextVectorizer()
        self.vectorizer.load(self.vectorizer_path)

    def preprocess_path(self, path):
        try:
            # Decode URL-encoded characters
            decoded_path = unquote(path)
            print(f"Decoded Path: {decoded_path}")  # Debugging

            # Extract the last segment of the path
            last_segment = decoded_path.split('/')[-1]
            print(f"Last Segment: {last_segment}")  # Debugging

            # Convert to a format suitable for the model
            if self.vectorizer:
                vectorized_path = self.vectorizer.transform([last_segment]).toarray()
                print(f"Vectorized Path: {vectorized_path}")  # Debugging

                # Check if the vectorized path is all zeros
                if not np.any(vectorized_path):
                    print("Warning: Vectorized path is all zeros. No meaningful tokens detected.")
                    return None
                
                return vectorized_path
            else:
                print("Vectorizer not loaded.")
                return None

        except Exception as e:
            print(f"Error during preprocessing: {e}")
            return None

    @abstractmethod
    def detect(self, path):
        pass

class SQLInjectionWAF(WAF):
    def detect(self, path):
        preprocessed_path = self.preprocess_path(path)
        if preprocessed_path is None:
            return False  # No meaningful tokens, assume no SQL injection

        try:
            prediction = self.model.predict(preprocessed_path) if self.model else [0]
            print(f"Prediction: {prediction}")  # Debugging
            return prediction[0] == 1  # Assuming 1 indicates SQL injection
        except Exception as e:
            print(f"Error during prediction: {e}")
            return False