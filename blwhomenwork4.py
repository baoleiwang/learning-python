import gensim
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter
from torch.utils.data import TensorDataset,DataLoader


def build_word2id(file, save_to_path=None):
    """
    :param file: word2id保存地址
    :param save_to_path: 保存训练语料库中的词组对应的word2vec到本地
    :return: None
    """
    word2id = {'_PAD_': 0}
    path = ['./Dataset/train.txt', './Dataset/validation.txt']

    for _path in path:
        with open(_path, encoding='utf-8') as f:
            for line in f.readlines():
                sp = line.strip().split()
                for word in sp[1:]:
                    if word not in word2id.keys():
                        word2id[word] = len(word2id)
    if save_to_path:
        with open(file, 'w', encoding='utf-8') as f:
            for w in word2id:
                f.write(w + '\t')
                f.write(str(word2id[w]))
                f.write('\n')

    return word2id

word2id = build_word2id('./Dataset/word2id.txt')
print(word2id)
def build_word2vec(fname, word2id, save_to_path=None):
    """
    :param fname: 预训练的word2vec.
    :param word2id: 语料文本中包含的词汇集.
    :param save_to_path: 保存训练语料库中的词组对应的word2vec到本地
    :return: 语料文本中词汇集对应的word2vec向量{id: word2vec}.
    """
    n_words = max(word2id.values()) + 1
    model = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=True)
    word_vecs = np.array(np.random.uniform(-1., 1., [n_words, model.vector_size]))
    for word in word2id.keys():
        try:
            word_vecs[word2id[word]] = model[word]
        except KeyError:
            pass
    if save_to_path:
        with open(save_to_path, 'w', encoding='utf-8') as f:
            for vec in word_vecs:
                vec = [str(w) for w in vec]
                f.write(' '.join(vec))
                f.write('\n')
    return word_vecs

word2vec = build_word2vec('./Dataset/wiki_word2vec_50.bin', word2id)
assert word2vec.shape == (58954, 50)
print(word2vec)
def cat_to_id(classes=None):
    """
    :param classes: 分类标签；默认为0:pos, 1:neg
    :return: {分类标签：id}
    """
    if not classes:
        classes = ['0', '1']
    cat2id = {cat: idx for (idx, cat) in enumerate(classes)}
    return classes, cat2id

def load_corpus(path, word2id, max_sen_len=50):
    """
    :param path: 样本语料库的文件
    :return: 文本内容contents，以及分类标签labels(onehot形式)
    """
    _, cat2id = cat_to_id()
    contents, labels = [], []
    with open(path, encoding='utf-8') as f:
        for line in f.readlines():
            sp = line.strip().split()
            label = sp[0]
            content = [word2id.get(w, 0) for w in sp[1:]]
            content = content[:max_sen_len]
            if len(content) < max_sen_len:
                content += [word2id['_PAD_']] * (max_sen_len - len(content))
            labels.append(label)
            contents.append(content)
    counter = Counter(labels)
    print('总样本数为：%d' % (len(labels)))
    print('各个类别样本数如下：')
    for w in counter:
        print(w, counter[w])

    contents = np.asarray(contents)
    labels = np.array([cat2id[l] for l in labels])

    return contents, labels
print('train corpus load: ')
train_contents, train_labels = load_corpus('./Dataset/train.txt', word2id, max_sen_len=50)
print('\nvalidation corpus load: ')
val_contents, val_labels = load_corpus('./Dataset/validation.txt', word2id, max_sen_len=50)
print('\ntest corpus load: ')
test_contents, test_labels = load_corpus('./Dataset/test.txt', word2id, max_sen_len=50)

class CONFIG():
    update_w2v = True           # 是否在训练中更新w2v
    vocab_size = 58954          # 词汇量，与word2id中的词汇量一致
    n_class = 2                 # 分类数：分别为pos和neg
    embedding_dim = 50          # 词向量维度
    drop_keep_prob = 0.5        # dropout层，参数keep的比例
    num_filters = 256           # 卷积层filter的数量
    kernel_size = 3             # 卷积核的尺寸
    pretrained_embed = word2vec # 预训练的词嵌入模型


class TextCNN(nn.Module):
    def __init__(self, config):
        super(TextCNN, self).__init__()
        update_w2v = config.update_w2v
        vocab_size = config.vocab_size
        n_class = config.n_class
        embedding_dim = config.embedding_dim
        num_filters = config.num_filters
        kernel_size = config.kernel_size
        drop_keep_prob = config.drop_keep_prob
        pretrained_embed = config.pretrained_embed

        # 使用预训练的词向量
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embed))
        self.embedding.weight.requires_grad = update_w2v
        # 卷积层
        self.conv = nn.Conv2d(1, num_filters, (kernel_size, embedding_dim))
        # Dropout
        self.dropout = nn.Dropout(drop_keep_prob)
        # 全连接层
        self.fc = nn.Linear(num_filters, n_class)

    def forward(self, x):
        x = x.to(torch.int64)
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = F.relu(self.conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        x = self.dropout(x)
        x = self.fc(x)
        return x

config = CONFIG()          # 配置模型参数
learning_rate = 0.001      # 学习率
batch_size = 32            # 训练批量
epochs = 4                 # 训练轮数
model_path = None          # 预训练模型路径
verbose = True             # 打印训练过程
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
contents = np.vstack([train_contents, val_contents])
labels = np.concatenate([train_labels, val_labels])

# 加载训练用的数据
train_dataset = TensorDataset(torch.from_numpy(contents).type(torch.float),
                              torch.from_numpy(labels).type(torch.long))
train_dataloader = DataLoader(dataset = train_dataset, batch_size = batch_size,
                              shuffle = True, num_workers = 0)


def train(dataloader):
    # 配置模型，是否继续上一次的训练
    model = TextCNN(config)
    if model_path:
        model.load_state_dict(torch.load(model_path))
    model.to(device)

    # 设置优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 设置损失函数
    criterion = nn.CrossEntropyLoss()

    # 定义训练过程
    for epoch in range(epochs):
        for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            output = model(batch_x)
            loss = criterion(output, batch_y)

            if batch_idx % 200 == 0 & verbose:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch + 1, batch_idx * len(batch_x), len(dataloader.dataset),
                    100. * batch_idx / len(dataloader), loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 保存模型
    torch.save(model.state_dict(), 'model.pth')
train(train_dataloader)
# 设置超参数
model_path = 'model.pth'   # 模型路径
batch_size = 32            # 测试批量
 # 加载测试集
test_dataset = TensorDataset(torch.from_numpy(test_contents).type(torch.float),
                            torch.from_numpy(test_labels).type(torch.long))
test_dataloader = DataLoader(dataset = test_dataset, batch_size = batch_size,
                            shuffle = False, num_workers = 0)


def predict(dataloader):
    # 读取模型
    model = TextCNN(config)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)

    # 测试过程
    count, correct = 0, 0
    for _, (batch_x, batch_y) in enumerate(dataloader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        output = model(batch_x)
        correct += (output.argmax(1) == batch_y).float().sum().item()
        count += len(batch_x)

    # 打印准确率
    print('test accuracy is {:.2f}%.'.format(100 * correct / count))

predict(test_dataloader)