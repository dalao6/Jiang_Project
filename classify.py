import re
import os
from jieba import cut
from itertools import chain
from collections import Counter
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report


def get_words(filename):
    """读取文本并过滤无效字符和长度为1的词"""
    words = []
    try:
        with open(filename, 'r', encoding='utf-8') as fr:
            for line in fr:
                line = line.strip()
                # 过滤无效字符
                line = re.sub(r'[.【】0-9、——。，！~\*]', '', line)
                # 使用jieba.cut()方法对文本切词处理
                line = cut(line)
                # 过滤长度为1的词
                line = filter(lambda word: len(word) > 1, line)
                words.extend(line)
    except FileNotFoundError:
        print(f"文件 {filename} 未找到，请检查路径。")
    return words


def get_top_words(top_num):
    """遍历邮件建立词库后返回出现次数最多的词"""
    all_words = []
    filename_list = [f'邮件_files/{i}.txt' for i in range(151)]
    # 遍历邮件建立词库
    for filename in filename_list:
        all_words.append(get_words(filename))
    # itertools.chain()把all_words内的所有列表组合成一个列表
    # collections.Counter()统计词个数
    freq = Counter(chain(*all_words))
    return [i[0] for i in freq.most_common(top_num)]


def build_vector_high_freq():
    """构建高频词特征向量"""
    top_words = get_top_words(100)
    all_words = []
    filename_list = [f'邮件_files/{i}.txt' for i in range(151)]
    for filename in filename_list:
        all_words.append(get_words(filename))
    vector = []
    for words in all_words:
        word_map = list(map(lambda word: words.count(word), top_words))
        vector.append(word_map)
    return np.array(vector)


def build_vector_tfidf():
    """构建TF-IDF特征向量"""
    filename_list = [f'邮件_files/{i}.txt' for i in range(151)]
    corpus = []
    for filename in filename_list:
        words = get_words(filename)
        corpus.append(' '.join(words))
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(corpus), vectorizer


def build_feature_vector(feature_method='high_freq', top_num=100):
    """根据特征方法构建特征向量"""
    if feature_method == 'high_freq':
        return build_vector_high_freq()
    elif feature_method == 'tfidf':
        vector, vectorizer = build_vector_tfidf()
        return vector, vectorizer
    else:
        raise ValueError("不支持的特征方法，可选 'high_freq' 或 'tfidf'")


def train_model(feature_method='high_freq', top_num=100, oversampling=False):
    """训练模型（支持过采样）"""
    labels = np.array([1] * 127 + [0] * 24)  # 127封垃圾邮件，24封普通邮件
    if feature_method == 'high_freq':
        vector = build_feature_vector(feature_method, top_num)
    elif feature_method == 'tfidf':
        vector, vectorizer = build_feature_vector(feature_method, top_num)

    # 过采样处理
    if oversampling:
        if feature_method == 'tfidf':
            vector = vector.toarray()
        sm = SMOTE(random_state=42)
        vector_res, labels_res = sm.fit_resample(vector, labels)
        model = MultinomialNB().fit(vector_res, labels_res)
    else:
        model = MultinomialNB().fit(vector, labels)

    if feature_method == 'tfidf':
        return model, vectorizer
    return model


def predict(model, filename, feature_method='high_freq', vectorizer=None):
    """对邮件进行分类预测"""
    words = get_words(filename)
    if feature_method == 'high_freq':
        top_words = get_top_words(100)
        current_vector = np.array(
            tuple(map(lambda word: words.count(word), top_words)))
        result = model.predict(current_vector.reshape(1, -1))
    elif feature_method == 'tfidf':
        current_vector = vectorizer.transform([' '.join(words)])
        result = model.predict(current_vector)
    return '垃圾邮件' if result == 1 else '普通邮件'


if __name__ == "__main__":
    labels = np.array([1] * 127 + [0] * 24)

    # 高频词特征模式
    print('高频词特征模式:')
    model_high_freq = train_model(feature_method='high_freq', oversampling=True)
    test_files = [f'邮件_files/{i}.txt' for i in range(151, 156)]
    for file in test_files:
        result = predict(model_high_freq, file, feature_method='high_freq')
        print(f"{os.path.basename(file)} 分类结果: {result}")

    # 评估高频词特征模式
    vector_high_freq = build_vector_high_freq()
    y_pred_high_freq = model_high_freq.predict(vector_high_freq)
    print("\n高频词特征模式分类评估报告:")
    print(classification_report(labels, y_pred_high_freq, target_names=['普通邮件', '垃圾邮件']))

    # TF-IDF特征模式
    print('\nTF-IDF特征模式:')
    model_tfidf, vectorizer = train_model(feature_method='tfidf', oversampling=True)
    for file in test_files:
        result = predict(model_tfidf, file, feature_method='tfidf', vectorizer=vectorizer)
        print(f"{os.path.basename(file)} 分类结果: {result}")

    # 评估TF-IDF特征模式
    vector_tfidf, _ = build_vector_tfidf()
    y_pred_tfidf = model_tfidf.predict(vector_tfidf)
    print("\nTF-IDF特征模式分类评估报告:")
    print(classification_report(labels, y_pred_tfidf, target_names=['普通邮件', '垃圾邮件']))
