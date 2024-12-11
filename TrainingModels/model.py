from sklearn.metrics import accuracy_score, classification_report
import joblib
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
# ensemble learning
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, StackingClassifier


def custom_tokenizer(text):
    tokens = text.split()
    return tokens

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'custom_tokenizer':
            return custom_tokenizer
        return super().find_class(module, name)


class TextVectorizer:
    def __init__(self):
        self.min_df = 1
        self.ngram_range = (1, 3)
        self.vectorizer = None

    def fit(self, data):
        if self.vectorizer is None:
            self.vectorizer = CountVectorizer(min_df=self.min_df, tokenizer=custom_tokenizer, ngram_range=self.ngram_range)
        X = self.vectorizer.fit_transform(data)
        return X

    def transform(self, data):
        return self.vectorizer.transform(data)

    def save(self, save_path):
        if self.vectorizer is not None:
            joblib.dump(self.vectorizer, save_path)

    def load(self, load_path):
        try:
            with open(load_path, 'rb') as f:
                self.vectorizer = CustomUnpickler(f).load()
            print("Vectorizer loaded successfully.")
        except Exception as e:
            print(f"Error loading vectorizer: {e}")
            self.vectorizer = None


class ClassifyModel:
    def __init__(self):
        self.classifier = None

    def fit(self, X_train, y_train):
        self.classifier.fit(X_train, y_train)

    def predict(self, X_test):
        return self.classifier.predict(X_test)

    def accuracy(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(accuracy)

    def report(self, X_test, y_test):
        y_pred = self.predict(X_test)
        report = classification_report(y_test, y_pred)
        print(report)

    def save(self, save_path):
        joblib.dump(self.classifier, save_path)

    def load(self, load_path):
        try:
            self.classifier = joblib.load(load_path)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.classifier = None


class SvmClassifier(ClassifyModel):
    def __init__(self, mode):
        super().__init__()
        if mode == "train":
            self.classifier = SVC()