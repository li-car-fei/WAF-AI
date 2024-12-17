from sklearn.metrics import accuracy_score, classification_report
import joblib
import pickle
import numpy as np
from gensim.models import Word2Vec
import os
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

# 一段文本转为一个固定长度的向量矩阵，用 Word2Vec 模型，长度进行截短、补零
# 使用 SVC 分类时，依然要铺平为一维向量，不能输入矩阵进行分类
class Word2VecVectorizer:
    def __init__(self, model_path=None, fixed_length=50):
        # 初始化参数
        self.fixed_length = fixed_length
        self.model = None

        # 如果提供了模型路径，则加载模型
        if model_path is not None and os.path.exists(model_path):
            self.load(model_path)

    def fit(self, documents):
        # 训练 Word2Vec 模型
        tokenized_docs = [custom_tokenizer(doc) for doc in documents]  # 使用简单分词
        self.model = Word2Vec(sentences=tokenized_docs, vector_size=100, window=5, min_count=1, workers=4)

        documents_vector_matrix = []
        for document in documents:
            documents_vector_matrix.append(self.transform_kernel(document))
        return documents_vector_matrix

    def transform(self, documents):
        matrix = []
        for document in documents:
            matrix.append(self.transform_kernel(document))
        return matrix

    def transform_kernel(self, text):
        # 将文本转换为固定长度的向量矩阵
        tokens = custom_tokenizer(text)
        vector_matrix = np.zeros((self.fixed_length, self.model.vector_size))  # 用零填充

        for i in range(self.fixed_length):
            if i < len(tokens) and tokens[i] in self.model.wv:
                vector_matrix[i] = self.model.wv[tokens[i]]

        # 二维铺平一维返回, 未转回 array
        return vector_matrix.flatten()

    def save(self, model_path):
        # 保存模型参数
        if self.model is not None:
            self.model.save(model_path)

    def load(self, model_path):
        # 加载模型参数
        if os.path.exists(model_path):
            self.model = Word2Vec.load(model_path)
        else:
            raise FileNotFoundError(f"Model file '{model_path}' not found.")


# 一段文本转换成一个固定长度的向量，长度是词典长度，即每个词的计数
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