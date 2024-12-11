import pandas as pd
import os
from sklearn.model_selection import train_test_split
from TrainingModels.model import TextVectorizer, SvmClassifier


def read_data(parent_dir):

    df_path = os.path.join(parent_dir, 'data', 'processed', 'sqli-by-chou@ibcher+.csv')
    df = pd.read_csv(df_path, usecols=['payload', 'is_malicious', 'injection_type'])

    print('Data distribution')
    print(df['injection_type'].value_counts())

    return df_path, df

def train():

    current_dir = os.getcwd()
    parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
    print('Current directory:', parent_dir)

    df_path, df = read_data(parent_dir)

    # data preprocessing
    df['payload'] = df['payload'].astype(str).fillna('')
    df = df[df['payload'] != '']
    df = df.drop_duplicates(subset=['payload'])

    # n-gram preprocessing
    count_vectorizer = TextVectorizer()
    X = count_vectorizer.fit(df['payload'])

    # train/test
    y = df['is_malicious']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # train model
    svc = SvmClassifier(mode="train")
    svc.fit(X_train, y_train)
    svc.accuracy(X_test, y_test)
    svc.report(X_test, y_test)

    # save model params
    count_vectorizer.save(os.path.join(parent_dir, 'WAF', 'models', 'svc_vc.pkl'))
    svc.save(os.path.join(parent_dir, 'WAF', 'models', 'svc.pkl'))

def test():
    sql_injections = [
        'verve',
        'helllo chouaib',
        'username',
        'password',
        'bounjour',
        "1' OR '1'='1",
        "1' OR '1'='1' --",
        "1' OR '1'='1' ({",
        "1' OR '1'='1' /*",
        "1' OR '1'='1' #",
        "1' OR '1'='1' AND '1'='1",
        "1' OR '1'='1' AND '1'='2",
        "1' OR '1'='1' UNION SELECT NULL, NULL",
        "1' OR '1'='1' UNION SELECT username, password FROM users",
        "1' OR '1'='1' UNION SELECT table_name, column_name FROM information_schema.columns"
    ]

    current_dir = os.getcwd()
    parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
    print('Current directory:', parent_dir)

    count_vectorizer = TextVectorizer()
    count_vectorizer.load(os.path.join(parent_dir, 'WAF', 'models', 'svc_vc.pkl'))
    svc = SvmClassifier(mode="test")
    svc.load(os.path.join(parent_dir, 'WAF', 'models', 'svc.pkl'))

    sql_injections_vectorized = count_vectorizer.transform(sql_injections).toarray()
    predictions = svc.predict(sql_injections_vectorized)

    for i, sql in enumerate(sql_injections):
        print(f"SQL Injection: {sql} -> Prediction: {predictions[i]}")


if __name__ == "__main__":

    train()
    test()




