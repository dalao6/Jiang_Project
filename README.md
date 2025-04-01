# README

## 代码核心功能说明

### 算法基础
本仓库采用多项式朴素贝叶斯分类器（Multinomial Naive Bayes）对邮件进行分类，判断邮件是垃圾邮件还是普通邮件。

#### 条件概率的特征独立性假设
多项式朴素贝叶斯分类器基于一个重要的假设：特征之间是相互独立的。在邮件分类的场景中，每个词都被视为一个特征，假设每个词的出现与否和其他词的出现是相互独立的。也就是说，一个词在邮件中出现的概率不受其他词是否出现的影响。

#### 贝叶斯定理在邮件分类中的具体应用形式
贝叶斯定理的公式为：
$$P(y|x) = \frac{P(x|y) \cdot P(y)}{P(x)}$$


在邮件分类中，我们要计算邮件属于垃圾邮件（\(C_1\)）和普通邮件（\(C_0\)）的概率，即 \(P(C_1|X)\) 和 \(P(C_0|X)\)，然后比较这两个概率的大小，将邮件分类到概率较大的类别中。由于 \(P(X)\) 对于所有类别都是相同的，因此在比较时可以忽略，只需要比较 \(P(X|C)P(C)\) 的大小。

### 数据处理流程

#### 分词处理
使用 `jieba` 库对邮件文本进行分词处理。在 `get_words` 函数中，通过 `jieba.cut()` 方法将文本分割成单个的词。

#### 停用词过滤
在 `get_words` 函数中，使用正则表达式过滤掉无效字符，如 `[.【】0-9、——。，！~\*]`，然后过滤掉长度为 1 的词，以去除一些无意义的单字。

### 特征构建过程

#### 高频词特征选择
高频词特征选择是指选择出现次数最多的前 \(n\) 个词作为特征。在代码中，通过 `get_top_words` 函数遍历所有邮件，统计每个词的出现次数，然后选择出现次数最多的前 100 个词作为特征。

- **数学表达形式**：设 \(f_i\) 表示第 \(i\) 个词的出现频率，选择 \(f_i\) 最大的前 \(n\) 个词作为特征。
- **实现差异**：在代码中，使用 `collections.Counter` 统计词的出现次数，然后通过 `most_common` 方法选择出现次数最多的前 \(n\) 个词。

#### TF-IDF特征加权
TF-IDF（Term Frequency-Inverse Document Frequency）是一种常用的特征加权方法，它结合了词的频率（TF）和逆文档频率（IDF）。

- **数学表达形式**：
  - 词频（TF）：\(TF_{i,j}=\frac{n_{i,j}}{\sum_{k}n_{k,j}}\)，其中 \(n_{i,j}\) 是词 \(i\) 在文档 \(j\) 中出现的次数，\(\sum_{k}n_{k,j}\) 是文档 \(j\) 中所有词的出现次数之和。
  - 逆文档频率（IDF）：\(IDF_i=\log\frac{|D|}{|j:t_i\in d_j|}\)，其中 \(|D|\) 是文档总数，\(|j:t_i\in d_j|\) 是包含词 \(i\) 的文档数。
  - TF-IDF：\(TF - IDF_{i,j}=TF_{i,j}\times IDF_i\)
- **实现差异**：在代码中，需要计算每个词在每个文档中的 TF 和 IDF，然后将它们相乘得到 TF-IDF 值。可以使用 `sklearn` 库中的 `TfidfVectorizer` 来实现。

## 高频词/TF-IDF两种特征模式的切换方法

### 高频词特征模式
当前代码使用的是高频词特征模式。如果要继续使用该模式，不需要进行任何修改。

### TF-IDF特征模式
要切换到 TF-IDF 特征模式，可以按照以下步骤进行修改：

```python
import re
import os
from jieba import cut
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def get_words(filename):
    """读取文本并过滤无效字符和长度为1的词"""
    words = []
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
    return ' '.join(words)

# 读取所有邮件文本
filename_list = ['邮件_files/{}.txt'.format(i) for i in range(151)]
corpus = []
for filename in filename_list:
    corpus.append(get_words(filename))

# 使用TfidfVectorizer进行特征提取
vectorizer = TfidfVectorizer()
vector = vectorizer.fit_transform(corpus)

# 0-126.txt为垃圾邮件标记为1；127-151.txt为普通邮件标记为0
labels = np.array([1]*127 + [0]*24)

model = MultinomialNB()
model.fit(vector, labels)

def predict(filename):
    """对未知邮件分类"""
    # 构建未知邮件的词向量
    words = get_words(filename)
    current_vector = vectorizer.transform([words])
    # 预测结果
    result = model.predict(current_vector)
    return '垃圾邮件' if result == 1 else '普通邮件'
# 使用SMOTE过采样（默认关闭）
model = train_model(feature_method='tfidf', oversampling=True)
print('151.txt分类情况:{}'.format(predict('邮件_files/151.txt')))
print('152.txt分类情况:{}'.format(predict('邮件_files/152.txt')))
print('153.txt分类情况:{}'.format(predict('邮件_files/153.txt')))
print('154.txt分类情况:{}'.format(predict('邮件_files/154.txt')))
print('155.txt分类情况:{}'.format(predict('邮件_files/155.txt')))
```

通过以上修改，代码将使用 TF-IDF 特征模式进行邮件分类。

#高频词特征模式分类评估报告:
             			 precision    recall      f1-score   support

        普通邮件      	   0.60      	   1.00          0.75         24
        垃圾邮件       	   1.00          0.87          0.93        127
    	accuracy                              0.89           151
   	macro avg       	   0.80          0.94          0.84        151
	weighted avg       0.94          0.89          0.90        151


#TF-IDF特征模式:
151.txt 分类结果: 垃圾邮件
152.txt 分类结果: 垃圾邮件
153.txt 分类结果: 普通邮件
154.txt 分类结果: 垃圾邮件
155.txt 分类结果: 普通邮件

#TF-IDF特征模式分类评估报告:
              			    precision    	recall  	f1-score   	    support

        普通邮件       		1.00      	1.00      	   1.00           24
        垃圾邮件       		1.00      	1.00      	   1.00          127
	accuracy                           				  1.00             151
        macro avg                 1.00         1.00      	  1.00       	     151
	weighted avg            1.00         1.00            1.00             151


进程已结束,退出代码0
