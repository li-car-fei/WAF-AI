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
import torch
import torch.nn as nn
from torch.utils.data import Dataset


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

    # 训练模型 => 一维向量 => 训练svm
    def fit(self, documents):
        # 训练 Word2Vec 模型
        tokenized_docs = [custom_tokenizer(doc) for doc in documents]  # 使用简单分词
        self.model = Word2Vec(sentences=tokenized_docs, vector_size=100, window=5, min_count=1, workers=4)

        documents_vector_matrix = []
        for document in documents:
            documents_vector_matrix.append(self.transform_kernel(document))
        return documents_vector_matrix

    # 获取一维向量 => 测试svm
    def transform(self, documents):
        matrix = []
        for document in documents:
            matrix.append(self.transform_kernel(document))
        return matrix

    # 单个文本 => 一维向量
    def transform_kernel(self, text):
        # 将文本转换为固定长度的向量矩阵
        tokens = custom_tokenizer(text)
        vector_matrix = np.zeros((self.fixed_length, self.model.vector_size))  # 用零填充

        for i in range(self.fixed_length):
            if i < len(tokens) and tokens[i] in self.model.wv:
                vector_matrix[i] = self.model.wv[tokens[i]]

        # 二维铺平一维返回, 未转回 array
        return vector_matrix.flatten()

    # 获取词典，用于生成模型的数据预处理
    def get_dict(self):
        vocab = self.model.wv.key_to_index
        word2idx = {word : i for i, word in enumerate(vocab)}
        idx2word = {i : word for word, i in word2idx.items()}
        word2idx['<PAD>'] = len(word2idx)
        idx2word[len(idx2word)] = '<PAD>'
        return word2idx, idx2word

    # 获取 sel_len * vec 数据 => 训练生成模型
    def get_generate_dataset(self, documents):
        data = []
        for document in documents:
            words = custom_tokenizer(document)
            words_matrix = [self.model.wv[word] for word in words if word in self.model.wv]

            # 小于 seq_length, 用 '<PAD>' 填充
            PAD_VECTOR = [0] * self.model.vector_size
            while len(words_matrix) < self.fixed_length:
                words_matrix.append(PAD_VECTOR)

            # 整理训练数据, x => y
            for i in range(len(words_matrix) - self.fixed_length):
                data.append((words_matrix[i:i + self.fixed_length], words_matrix[i + self.fixed_length]))

        return data

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

# 训练生成模型数据集类
class Word2VecDataset(Dataset):
    def __init__(self, documents, word2vec_vectorizer, fix_length=50):
        self.word2vec_vectorizer = word2vec_vectorizer
        self.fix_length = fix_length
        self.data = self.word2vec_vectorizer.get_generate_dataset(documents)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# lstm 文本生成模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.permute(1, 0, 2)  # (seq_length, batch_size, input_size)
        out, _ = self.lstm(x)
        out = self.fc(out[-1])
        return out


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