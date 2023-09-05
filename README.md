# Sentiment-Analysis
UCAS文本数据挖掘作业

# **情感分析报告**

文本情感分析主要应用于用户情感信息获取、舆情控制、产品推荐等方面，是指对带有情感色彩的主观性文本进行采集、处理、分析、归纳和推理的过程，涉及到人工智能、机器学习、数据挖掘、自然语言处理等多个研究领域。
根据使用的不同方法，将其划分为基于情感词典的规则方法、传统的统计学习方法、基于深度学习的情感分类方法。通过对这三种方法进行对比，并对不同方法的优缺点进行归纳。

## **1 研究进展**

## **2 实现过程**

### 2.1 数据集选取

本实验数据集选取中文微博数据集 `weibo_senti_100k`，内含10万余条带情感标注的新浪微博评论，正负向评论约各5万条。数据集包含两个字段：label字段中1表示正向评论，0表示负向评论；review字段中内容为微博评论内容<img src="image/%E6%88%AA%E5%B1%8F2023-06-13%2010.18.39.png" alt="截屏2023-06-13 10.18.39" style="zoom:50%;" />

本实验从10万条数据中随机抽取1万条进行实验

### 2.2 数据预处理

处理数据包括一下步骤：

1. 数据清洗：去除用户名、特殊字符以及非中文字符串（如网站链接）等无用信息
2. 分词处理：使用jieba库进行分词
3. 去除停用词

```python
def data_cut(self, review):
    index = 0
    review_cut = []
    for item in review:
        # 删除用户名
        item = re.sub('@.*?:', '', item)
        item = re.sub('@.*?：', '', item)
        # 删除特殊字符
        item = re.sub(r'\W+', ' ', item).replace('_', ' ')
        # 分词
        cut = jieba.lcut(item)
        segResult = []
        # 判断非中文字符串如链接等
        for word in cut:
            if ('\u4e00' <= word <= '\u9fa5'):
                segResult.append(word)
        review_cut.append(' '.join(segResult))
        index += 1
    return review_cut
```

### **2.3 基于情感词典进行情感分析**

#### 2.3.1 **加载词典**

- 本实验选取两种类型的词典。
  1. BonsonNLP情感词典：该情感词典包含了一系列中文词汇以及它们的情感强度分值，范围是0到1，表示该词汇在情感方面的程度。例如，一个词汇的情感分值为 0.8，表示该词汇在情感方面非常积极。
  2. 台湾大学NTUSD简体中文情感词典：该词典划分为正向情感词汇 `ntusd-positive.txt`和负向情感词汇 `ntusd-negative.txt`
- 加载其他预定义的词典，包括程度副词词典，否定词词典，以及停用词词典。
  - 程度副词词典来自于知网HowNet，该词典将程度副词分为六个级别，并分别赋值1.8、1.6、1.5、0.8、0.7、0.5，用于后续情感得分计算。
  - 停用词词典使用哈工大停用词表

```python
def load_dict(self, request):
    path = './dictionary/'
    if request == 'degree':
        degree_file = open(path + 'degree.txt', 'r+')
        degree_list = degree_file.readlines()
        degree_dict = defaultdict()
        for i in degree_list:
            degree_dict[i.split(',')[0]] = float(i.split(',')[1])
        return degree_dict
    elif request == 'sentiment':
        with open(path + 'sentiment_score.txt', 'r+') as f:
            sen_list = f.readlines()
            sen_dict = defaultdict()
            for i in sen_list:
                if len(i.split(' ')) == 2:
                    sen_dict[i.split(' ')[0]] = float(i.split(' ')[1])
        return sen_dict
    elif request == "positive":
        file = open(path + "positive_simplified.txt")
    elif request == "negative":
        file = open(path + "negative_simplified.txt")
    elif request == "inverse":
        file = open(path + "inverse_words.txt")
    elif request == 'stopwords':
        file = open(path + 'stopwords.txt')
    else:
        return None
    dict_list = []
    for word in file:
        dict_list.append(word.strip())
    return dict_list
  
if dict_choice == 'Boson':
    self.sentiment_dict = self.load_dict('sentiment')
elif dict_choice == 'NTUSD':
    self.positive_words = self.load_dict('positive')
    self.negative_words = self.load_dict('negative')
```

#### 2.3.2 **情感分类**

> 根据情感词典，程度副词，否定词等对文本进行情感分类。对于每个词，如果它在正面情感词列表中，就增加分数；如果在负面情感词列表中，就减少分数。同时，对于前置的否定词和程度副词也会相应地调整分数。

由于程度副词位置的不同会产生不同的情感强度。所以结合程度副词、否定词词典，并根据否定词和程度副词的不同位置制定以下规则

（1）否定词+程度副词+情感词：$score=w_{sen}*(-1)*w_{adv}*0.5$

（2）程度副词+否定词+情感词：$score=w_{sen}*(-1)*w_{adv}*2$

$w_{sen}$表示情感词语的强度值。针对BosonNLP情感词典，将分词后的数据对应BosonNLP词典进行逐个匹配，$w_{sen}$可直接从词典中得到。针对NTUSD情感词典，$w_{sen}$表示为正向情感词+1，负向情感词-1

$w_{adv}$表示程度副词的权值，根据人工制定的程度副词极性表来确定。

最后统计计算分值总和，如果分值大于0，则情感倾向表示为积极，如果分值小于0，则表示情感倾向为消极。

基于BosonNLP情感词典的情感分类：

```python
def classify_words_pn(self, word_list): 
    z = 0  
    score = []  
    for word_index, word in enumerate(word_list):
        w = 0  
        if word in self.positive_words:  # 为正面情感词
            w += 1
            for i in range(z, int(word_index)):  
                if word_list[i] in self.inverse_words: 
                    w = w * (-1)
                    for j in range(z, i): 
                        if word_list[j] in self.degree_dict:
                            w = w * 2 * self.degree_dict[word_list[j]]
                            break
                    for j in range(i, int(word_index)): 
                        if word_list[j] in self.degree_dict:
                            w = w * 0.5 * self.degree_dict[word_list[j]]
                            break
                elif word_list[i] in self.degree_dict:
                    w = w * self.degree_dict[word_list[i]]
            z = int(word_index) + 1
        if word in self.negative_words:  # 为负面情感词
            w -= 1
            for i in range(z, int(word_index)):
                if word_list[i] in self.inverse_words:
                    w = w * (-1)
                    for j in range(z, i):  
                        if word_list[j] in self.degree_dict:
                            w = w * 2 * self.degree_dict[word_list[j]]
                            break
                    for j in range(i, int(word_index)): 
                        if word_list[j] in self.degree_dict:
                            w = w * 0.5 * self.degree_dict[word_list[j]]
                            break
                elif word_list[i] in self.degree_dict:
                    w *= self.degree_dict[word_list[i]]
            z = int(word_index) + 1
        score.append(w)
    score = sum(score)
    return score
```

基于NTUSD情感词典的情感分类：

```python
def classify_words_value(self, word_list):
    scores = []
    z = 0
    for word_index, word in enumerate(word_list):
        score = 0
        if word in self.sentiment_dict.keys() and word not in self.inverse_words and word not in self.degree_dict.keys():
            score = self.sentiment_dict[word]
            for i in range(z, int(word_index)): 
                if word_list[i] in self.inverse_words:  
                    score = score * (-1)
                    for j in range(z, i): 
                        if word_list[j] in self.degree_dict:
                            score = score * self.degree_dict[word_list[j]] * 2
                            break
                    for j in range(i, int(word_index)):
                        if word_list[j] in self.degree_dict:
                            score = score * self.degree_dict[word_list[j]] * 0.5
                            break
                elif word_list[i] in self.degree_dict:
                    score = score * float(self.degree_dict[word_list[i]])
            z = int(word_index) + 1
        scores.append(score)
    scores = sum(scores)
    return scores
```

#### 2.3.3 **性能评估**

使用实际的情感标签与预测的情感标签，计算准确率，精确率，召回率以及F1分数，作为情感分析模型的评估指标。

```python
def evaluate(self, label, predicts):
    accuracy = accuracy_score(label, predicts)
    precision = precision_score(label, predicts)
    recall = recall_score(label, predicts)
    f1 = f1_score(label, predicts)
    print('准确率：', accuracy, '\n正确率：', precision, '\n召回率：', recall, '\nF1值：', f1)
```

#### 2.3.4 实验结果

|                                | 准确率 | 精确率 | 召回率 |   F1值 |
| :----------------------------- | :----: | :----: | :----: | -----: |
| **BosonNLP词典**               | 0.7966 | 0.7396 | 0.9145 | 0.8178 |
| **NTUSD词典**                  | 0.7780 | 0.7609 | 0.8097 | 0.7845 |
| **BosonNLP词典（去除停用词）** | 0.7535 | 0.7092 | 0.8581 | 0.7766 |
| **NTUSD词典（去除停用词）**    | 0.7025 | 0.7352 | 0.6314 | 0.6793 |

#### 2.3.5 实验分析

由于数据集选取微博数据集，其中网络用语、表情等特殊词语对情感分类也会产生一定影响，选取的情感词典没有针对这些特殊词的分析，所以准确率不高。同时停用词的去除可能会去掉一些对情感分类有用的关键词，对情感分数产生影响。后续可以考虑选用更有针对性的情感词典进行分析。

基于情感词典分类的局限性：

（1）不同的词典对同一数据集的分类的结果不同，取决于词典中词的构成，范围，极性值等因素

（2）规则的固定性难以针对话语的具体环境灵活变通，同一规则在一批数据中难以适用于所有的文本

（3）易于区分积极、消极、中性情感，难以区分具体情感如：喜悦、愤怒、厌恶、悲伤等情感

（4）情感词典将常用词制定的分数唯一，一词多义难以分辨，对情感分类的准确率影响较大

### **2.4 基于机器学习的情感分类**

> **支持向量机** 是一种二元线性分类器，其基本原理是在高维空间中寻找一个超平面，使得它能够将不同类别的样本分隔开。这个超平面被选定为各类别之间的"最大间隔"边界，也就是说，它将尽量远离各类别的最近样本点。
>
> **朴素贝叶斯** 是基于贝叶斯定理的一种简单概率分类器，它假设特征之间是相互独立的。
>
> **决策树** 是一种树形结构的分类器，每个内部节点代表一个特征，每个分支代表一个决策规则，每个叶节点代表一个结果（类别）。创建决策树的过程就是寻找一组规则的过程，这组规则可以将数据分类的"最好"，也就是使信息增益或者其他评价指标最大。
>
> **随机森林** 是一种集成学习模型，它构建了多个决策树并结合他们的预测结果（通过投票或平均）来提高模型的整体预测性能。随机森林在创建决策树的过程中引入了随机性，这样可以降低模型的方差并防止过拟合。
>
> **Adaboost** 是一种自适应的集成学习算法，它在每轮迭代中增加一个新的弱分类器，这个弱分类器被训练为纠正之前所有分类器的预测错误。Adaboost通过在每轮迭代中增加那些被前一个分类器错误分类的样本的权重，并减少那些被正确分类的样本的权重，使得新的分类器更多地关注那些"难以分类"的样本，从而提高模型的预测精度。

#### 2.4.1 特征抽取

使用TF-IDF方法将文本信息转换为词向量，本实验直接调用sklearn库的api生成TF-IDF词向量

```python
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform((d for d in review_cut_list))
```

#### 2.4.2 情感分类

本实验使用sklearn中的分类器进行情感分析，选用SVM、朴素贝叶斯、决策树、随机森林、Adaboost分别进行实验，并计算准确率、精确率、召回率和F1分数进行实验评估

```python
def train_and_evaluate(classifier, X_train, y_train, X_test, y_test):
    model = classifier()
    model.fit(X_train, y_train)

    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='macro')
    recall = recall_score(y_test, predictions, average='macro')
    f1 = f1_score(y_test, predictions, average='macro')

    print(f"Model: {classifier.__name__}")
    print(f"Train score: {train_score:.4f}")
    print(f"Test score: {test_score:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

classifiers = [LinearSVC, MultinomialNB, DecisionTreeClassifier, RandomForestClassifier, AdaBoostClassifier]
for classifier in classifiers:
    SentimentAnalysis.train_and_evaluate(classifier, X_train, y_train, X_test, y_test)
```

#### 2.4.3 实验结果

 - 对于给定的分类器，对每个分类器进行训练并在测试集上进行评估。

|                | 准确率 | 精确率 | 召回率 |   F1值 |
| :------------- | :----: | :----: | :----: | -----: |
| **SVM**        | 0.8545 | 0.8551 | 0.8547 | 0.8545 |
| **朴素贝叶斯** | 0.8225 | 0.8246 | 0.8221 | 0.8221 |
| **决策树**     | 0.8810 | 0.8810 | 0.8810 | 0.8810 |
| **随机森林**   | 0.8935 | 0.8938 | 0.8934 | 0.8935 |
| **Adaboost**   | 0.8965 | 0.8967 | 0.8966 | 0.8965 |

#### 2.4.4 实验分析

基于机器学习的情感分类训练速度较快，但由于其依赖先验知识，对训练数据要求较高

### **2.5 基于深度神经网络的情感分类**

#### 2.5.0 实验框架与环境

使用 `pytorch` 框架训练神经网络

实验环境：NVIDIA GeForce GTX 1650 GPU/4G

#### 2.5.1 文本表示

首先将文本信息转换为向量信息。遍历文本中的所有词组，将每个词语映射成唯一一个整数索引，得到一个词语字典，字典中的数量为该数据集中所有出现过词语的数量。

```python
def compute_word2index(sentences, word2index):
    for sentence in sentences:
        for word in sentence.split():
            if word not in word2index:
                word2index[word] = len(word2index)
    return word2index
```

通过上述生成的词语字典，将句子中的每个词语映射到其在词语字典的索引，并将结果存储在一个列表中，并通过截断或末尾填充0操作，使每句话长度相等

```python
def compute_sent2index(sentence, max_len, word2index):
    sent2index = [word2index.get(word, 0) for word in sentence.split()]
    if len(sentence.split()) < max_len:
        sent2index += (max_len - len(sentence.split())) * [0]
    else:
        sent2index = sentence[:max_len]
    return sent2index
```

最后便可生成词向量和句向量

```python
def text_embedding(self, sentences):
    word2index = {"PAD": 0}
    word2index = self.compute_word2index(sentences, word2index)
    sent2indexs = []
    for sent in sentences:
        sentence = self.compute_sent2index(sent, MAX_LEN, word2index)
        sent2indexs.append(sentence)
    return word2index, sent2indexs
```

#### 2.5.2 网络选取

本次我们使用了两种神经网络用于情感分析  

1.TextCNN首先将输入的文本编码为词向量，然后使用卷积神经网络进行特征提取，最后通过全连接层进行分类。  

![截屏2023-06-13 12.45.15](image/%E6%88%AA%E5%B1%8F2023-06-13%2012.45.15.png)

```python
class TextCNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, feature_size, windows_size, max_len, n_class):
        super(TextCNN, self).__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.conv1 = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=embedding_dim, out_channels=feature_size, kernel_size=h),
                          nn.LeakyReLU(),
                          nn.MaxPool1d(kernel_size=max_len - h + 1),
                          )
            for h in windows_size]
        )
        self.dropout = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(in_features=feature_size * len(windows_size), out_features=n_class)

    def forward(self, x):
        x = self.embed(x)
        x = x.permute(0, 2, 1)
        x = [conv(x) for conv in self.conv1]
        x = torch.cat(x, 1)
        x = x.view(-1, x.size(1))
        x = self.dropout(x)
        x = self.fc1(x)
        return x
```

2.BiLSTM 是一个基于双向LSTM的文本分类模型，它首先将输入的文本编码为词向量，然后通过双向LSTM进行特征提取，最后通过全连接层进行分类。  

```python
class BiLSTM(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, num_layers, num_classes, device):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            bidirectional=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(in_features=hidden_size * 2, out_features=num_classes)

    def forward(self, x):
        x = self.embed(x)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        out = torch.cat([out[:, -1, :self.hidden_size], out[:, 0, self.hidden_size:]], dim=1)
        x = self.dropout(out)
        x = self.fc(x)
        return x
```

#### **2.5.3 参数设置**  

**批处理大小：**`batchsize=32`

**训练轮数:** `epoch=30`。

**滑动窗口大小：** `windowsize=[2,4,3]`

**文本最大长度：** `max_len=100` 

> 如果文本长度超过这个值，将其截断；如果文本长度小于100，补0填充。

**词向量维度：** `embedding_dim=200` 

**损失函数：** 使用交叉熵损失函数 `CrossEntropyLoss`

**优化器：** 使用 `Adam` 优化器，加入L2正则化减少过拟合

**学习率：** `learning_rate=0.0001`

#### 2.5.4 模型构建及训练

```python
def train(self, epochs, loss_func):
    train_len = len(self.train_loader.dataset)
    test_len = len(self.test_loader.dataset)

    for epoch in range(epochs):
        self.model.train()
        train_correct, test_correct, batch_num, train_loss, test_loss = 0, 0, 0, 0, 0
        for i, (input, label) in enumerate(self.train_loader):
            batch_num += 1
            input, label = input.to(self.device), label.to(self.device)
            output = self.model(input)
            loss = loss_func(output, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            pred = torch.max(output, 1)[1].cpu().numpy()
            label = label.cpu().numpy()
            correct = (pred == label).sum()
            train_correct += correct
            train_loss += loss.item()

        self.model.eval()
        with torch.no_grad():
            for ii, (input, label) in enumerate(self.test_loader):
                input, label = input.to(self.device), label.to(self.device)
                output = self.model(input)
                loss = loss_func(output, label)

                pred = torch.max(output, 1)[1].cpu().numpy()
                label = label.cpu().numpy()
                correct = (pred == label).sum()
                test_correct += correct
                test_loss += loss.item()

            train_acc = train_correct / train_len
            test_acc = test_correct / test_len

            print(f"Epoch: {epoch + 1}, train_loss: {train_loss / (i + 1):.5f}, test_loss: {test_loss / (ii + 1):.5f}, train_acc: {train_acc * 100:.2f}%, test_acc: {test_acc * 100:.2f}%")


model = TextCNN(vocab_size=len(word2index), embedding_dim=EMBEDDING_DIM, windows_size=WINDOWS_SIZE, max_len=MAX_LEN, feature_size=FEATURE_SIZE, n_class=N_CLASS).to(DEVICE)
model = BiLSTM(num_embeddings=len(word2index), embedding_dim=EMBEDDING_DIM, hidden_size=FEATURE_SIZE, num_layers=2, num_classes=N_CLASS, device=DEVICE).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-5)
trainer = Trainer(model, dataloader_train, dataloader_test, optimizer, DEVICE)

# 模型训练
trainer.train(EPOCHS, nn.CrossEntropyLoss())
```

#### 2.5.5 实验结果：

使用1万条数据，将80%用作训练集，20%用作测试集，训练结果如下

使用TextCNN训练结果：

<img src="image/%E6%88%AA%E5%B1%8F2023-06-13%2014.17.42.png" alt="截屏2023-06-13 14.17.42" style="zoom:50%;" />

使用BiLSTM训练结果：

<img src="image/%E6%88%AA%E5%B1%8F2023-06-13%2014.18.17.png" alt="截屏2023-06-13 14.18.17" style="zoom:50%;" />

**各训练10轮后，得到可用于预测的模型的性能指标如下：**  

|              | 准确率 | 精确率 | 召回率 |   F1值 |
| :----------- | :----: | :----: | :----: | -----: |
| **CNN**      | 0.9700 | 0.9700 | 0.9700 | 0.9700 |
| **双向LSTM** | 0.9565 | 0.9985 | 0.9985 | 0.9985 |

#### 2.5.6 实验分析

回顾前文用到的方法：  

|                | 准确率 | 精确率 | 召回率 |   F1值 |
| :------------- | :----: | :----: | :----: | -----: |
| **情感词典**   | 0.7966 | 0.7396 | 0.9145 | 0.8178 |
| **SVM**        | 0.8545 | 0.8551 | 0.8547 | 0.8545 |
| **朴素贝叶斯** | 0.8225 | 0.8246 | 0.8221 | 0.8221 |
| **决策树**     | 0.8810 | 0.8810 | 0.8810 | 0.8810 |
| **随机森林**   | 0.8935 | 0.8938 | 0.8934 | 0.8935 |
| **Adaboost**   | 0.8965 | 0.8967 | 0.8966 | 0.8965 |

我们可以看出，在我们测试的所有模型中，**深度学习模型（TextCNN和BiLSTM）** 在准确率，精确率，召回率和F1值上均超过了传统的机器学习模型（SVM、朴素贝叶斯、决策树、随机森林和Adaboost）和情感词典方法。  

特别是TextCNN，它在所有评估指标上都取得了最好的效果。这可能是因为卷积神经网络能够有效地学习文本的局部特征，并且可以同时考虑不同长度的单词组合，以捕捉到更丰富的语义信息。而双向LSTM在这个任务中的表现也非常出色，尤其在精确率和召回率方面达到了接近完美的结果，说明该模型对文本中的情感极性具有很强的识别能力。

## **结语**
