import random

import pandas as pd
import os
from TrainingModels.model import Word2VecVectorizer, Word2VecDataset, LSTMModel
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np



def read_data(parent_dir):

    df_path = os.path.join(parent_dir, 'data', 'processed', 'sqli-by-chou@ibcher+.csv')
    df = pd.read_csv(df_path, usecols=['payload', 'is_malicious', 'injection_type'])

    print('Data distribution')
    print(df['injection_type'].value_counts())

    return df_path, df

# 训练文本生成模型
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    current_dir = os.getcwd()
    parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
    print('Current directory:', parent_dir)

    # load word2vec 模型
    word2vec_vectorizer = Word2VecVectorizer(fixed_length=5)
    word2vec_vectorizer.load(os.path.join(parent_dir, 'WAF', 'models', 'svc_word2vec_vc.pkl'))
    word2idx, idx2word = word2vec_vectorizer.get_dict()

    # 读取原始训练数据
    df_path, df = read_data(parent_dir)
    df['payload'] = df['payload'].astype(str).fillna('')
    df = df[df['payload'] != '']
    df = df.drop_duplicates(subset=['payload'])
    df = df[df['is_malicious'] == 1]
    data = df['payload'].tolist()

    # data loader
    dataset = Word2VecDataset(data, word2vec_vectorizer, fix_length=5)
    print("Train data length: ", len(dataset))
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    # generate model
    model = LSTMModel(input_size=word2vec_vectorizer.model.vector_size, hidden_size=128, output_size=len(word2idx))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 20
    for epoch in range(epochs):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            model.zero_grad()
            output = model(x)
            loss = criterion(output, y.argmax(dim=1))  # 目标为索引
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

    model_path = os.path.join(parent_dir, 'TrainingModels', 'models', 'generate_lstm.pth')
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(model.state_dict(), model_path)

def inference(num=500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    current_dir = os.getcwd()
    parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
    print('Current directory:', parent_dir)

    # load word2vec 模型
    word2vec_vectorizer = Word2VecVectorizer(fixed_length=5)
    word2vec_vectorizer.load(os.path.join(parent_dir, 'WAF', 'models', 'svc_word2vec_vc.pkl'))
    word2idx, idx2word = word2vec_vectorizer.get_dict()

    # load lstm generate 模型
    model_path = os.path.join(parent_dir, 'TrainingModels', 'models', 'generate_lstm.pth')
    model = LSTMModel(input_size=word2vec_vectorizer.model.vector_size, hidden_size=128, output_size=len(word2idx))
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    def generate_sample(fix_length=5):

        # <PAD> 作为起始生成词输入
        start_word = '<PAD>'
        pad_vector = np.array([0] * word2vec_vectorizer.model.vector_size)
        input_vector = pad_vector.reshape(1, 1, -1)
        input_tensor = torch.tensor(input_vector).float().to(device)

        generated_text = []
        for _ in range(fix_length):
            with torch.no_grad():
                output = model(input_tensor)

            # 取最后一个时间步的输出
            last_output = output[-1, :]  # (batch_size, output_size)
            # 计算下一个单词的索引
            next_word_idx = torch.argmax(last_output, dim=0).item()
            next_word = idx2word[next_word_idx]  # 获取对应的词

            # 添加到生成文本中
            generated_text.append(next_word)

            if next_word == '<PAD>':
                input_vector = pad_vector.reshape(1, 1, -1)
            else:
                input_vector = word2vec_vectorizer.model.wv[next_word].reshape(1, 1, -1)

            input_tensor = torch.tensor(input_vector).float().to(device)

        return ' '.join(generated_text)

    # 保存地址
    generated_samples_path = os.path.join(parent_dir, 'data', 'raw', 'generated.txt')

    with open(generated_samples_path, 'w') as file:
        for _ in range(num):
            generate_text = generate_sample(fix_length=5)
            file.write(generate_text + '\n')

    print(f"Generated {num} samples and saved to {generated_samples_path}")

if __name__ == "__main__":

    # train()
    inference()