import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import random
import jieba

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 中文分词函数
def tokenize(text):
    return list(jieba.cut(text))

# 构建词汇表类
class Vocabulary:
    def __init__(self, min_freq=1):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}  # 特殊标记
        self.idx2word = {0: "<PAD>", 1: "<UNK>", 2: "<SOS>", 3: "<EOS>"}
        self.min_freq = min_freq

    def build_vocab(self, sentences):
        word_freq = Counter()
        for sentence in sentences:
            tokens = tokenize(sentence)
            word_freq.update(tokens)
        
        words = [word for word, freq in word_freq.items() if freq >= self.min_freq]
        for idx, word in enumerate(words, start=4):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def __len__(self):
        return len(self.word2idx)

# 数据集类
class ChatbotDataset(Dataset):
    def __init__(self, questions, answers, vocab):
        self.questions = questions
        self.answers = answers
        self.vocab = vocab

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        answer = self.answers[idx]
        question_indices = [self.vocab.word2idx.get(token, self.vocab.word2idx["<UNK>"]) for token in tokenize(question)]
        answer_indices = [self.vocab.word2idx["<SOS>"]] + \
                         [self.vocab.word2idx.get(token, self.vocab.word2idx["<UNK>"]) for token in tokenize(answer)] + \
                         [self.vocab.word2idx["<EOS>"]]
        return question_indices, answer_indices

# 数据加载器的 collate_fn
def collate_fn(batch):
    questions, answers = zip(*batch)
    question_tensors = [torch.tensor(q, dtype=torch.long) for q in questions]
    answer_tensors = [torch.tensor(a, dtype=torch.long) for a in answers]
    question_tensors = pad_sequence(question_tensors, batch_first=True, padding_value=0)
    answer_tensors = pad_sequence(answer_tensors, batch_first=True, padding_value=0)
    return question_tensors, answer_tensors

# 编码器
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, embedding_dim):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        x = self.embedding(x)
        out, hidden = self.lstm(x)
        return out, hidden

# 注意力机制
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, hidden, encoder_outputs):
        # hidden的形状是元组(h_n, c_n)，每个的形状是(num_layers * num_directions, batch_size, hidden_size)
        # 我们只需要最后一个层的隐藏状态
        h_n = hidden[0]  # (num_layers * num_directions, batch_size, hidden_size)
        last_hidden = h_n[-1]  # (batch_size, hidden_size)
        
        batch_size, seq_len, _ = encoder_outputs.size()
        
        # 扩展last_hidden以匹配encoder_outputs的时间步
        last_hidden = last_hidden.unsqueeze(1).expand(batch_size, seq_len, self.hidden_size)  # (batch_size, seq_len, hidden_size)
        
        # 计算注意力权重
        attn_energies = torch.tanh(self.attn(torch.cat((encoder_outputs, last_hidden), dim=2)))
        attn_energies = torch.sum(attn_energies * self.v, dim=2)  # (batch_size, seq_len)
        attn_weights = self.softmax(attn_energies)  # (batch_size, seq_len)
        
        # 计算上下文向量
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # (batch_size, 1, hidden_size)
        return context, attn_weights

# 解码器（带注意力机制）
class DecoderWithAttention(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, embedding_dim):
        super(DecoderWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim + hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.attention = Attention(hidden_size)

    def forward(self, x, hidden, encoder_outputs):
        x = self.embedding(x)  # (batch_size, 1, embedding_dim)
        context, attn_weights = self.attention(hidden, encoder_outputs)  # (batch_size, 1, hidden_size)
        x = torch.cat((x, context), dim=2)  # (batch_size, 1, embedding_dim + hidden_size)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden, attn_weights

# Seq2Seq 模型
import torch
import torch.nn as nn
import random

class ChatDataset(Dataset):
    def __init__(self, questions, answers, vocab):
        """
        初始化数据集
        :param questions: 问题列表
        :param answers: 答案列表 
        :param vocab: 词汇表对象
        """
        self.questions = questions
        self.answers = answers
        self.vocab = vocab

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        """
        获取单个训练样本
        返回: (问题token索引, 答案token索引)
        """
        question = self.questions[idx]
        answer = self.answers[idx]
        
        # 中文分词
        question_tokens = list(jieba.cut(question))
        answer_tokens = list(jieba.cut(answer))
        
        # 转换为索引，未知词用<UNK>表示
        question_indices = [
            self.vocab.word2idx.get(token, self.vocab.word2idx["<UNK>"]) 
            for token in question_tokens
        ]
        
        # 答案添加<SOS>和<EOS>
        answer_indices = (
            [self.vocab.word2idx["<SOS>"]] +
            [self.vocab.word2idx.get(token, self.vocab.word2idx["<UNK>"]) 
             for token in answer_tokens] +
            [self.vocab.word2idx["<EOS>"]]
        )
        
        return torch.tensor(question_indices, dtype=torch.long), \
               torch.tensor(answer_indices, dtype=torch.long)

def collate_fn(batch):
    questions, answers = zip(*batch)
    question_tensors = [torch.tensor(q, dtype=torch.long) for q in questions]
    answer_tensors = [torch.tensor(a, dtype=torch.long) for a in answers]
    question_tensors = pad_sequence(question_tensors, batch_first=True, padding_value=0)
    answer_tensors = pad_sequence(answer_tensors, batch_first=True, padding_value=0)
    return question_tensors, answer_tensors

def read_data(file_path):
    questions = []
    answers = []
    current_question = []
    current_answer = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line.startswith('？'):
                if current_question:
                    questions.append('\n'.join(current_question))
                    answers.append('\n'.join(current_answer))
                    current_question = []
                    current_answer = []
                current_question.append(line[1:].strip())
            elif line.startswith('！'):
                current_answer.append(line[1:].strip())
        if current_question:
            questions.append('\n'.join(current_question))
            answers.append('\n'.join(current_answer))
    
    return questions, answers

class Seq2SeqWithAttention(nn.Module):
    def __init__(self, encoder, decoder, max_length = 2000):
        super(Seq2SeqWithAttention, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.max_length = max_length

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(src.device)
        encoder_out, hidden = self.encoder(src)

        input = trg[:, 0].unsqueeze(1)  # (batch_size, 1)
        
        for t in range(1, trg_len):
            output, hidden, _ = self.decoder(input, hidden, encoder_out)
            outputs[:, t] = output.squeeze(1)
            
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(2)
            input = trg[:, t].unsqueeze(1) if teacher_force else top1

        return outputs

def generate_response(self, input_text):
    # 分词和转换为索引
    tokens = [self.vocab.word2idx.get(word, self.vocab.word2idx["<UNK>"]) for word in tokenize(input_text)]
    
    # 转换为张量
    input_tensor = torch.tensor([tokens], dtype=torch.long).to(self.device)
    
    # 创建虚拟目标张量，只包含 <SOS> 标记
    sos_token = torch.tensor([[self.vocab.word2idx["<SOS>"]]], dtype=torch.long).to(self.device)
    
    # 初始化响应列表
    response = []
    
    # 获取编码器的输出和隐藏状态
    encoder_out, hidden = self.model.encoder(input_tensor)
    
    # 使用 <SOS> 标记作为初始输入
    input = sos_token
    
    # 生成响应
    with torch.no_grad():
        for _ in range(self.model.max_length):
            output, hidden, _ = self.model.decoder(input, hidden, encoder_out)
            top1 = output.argmax(2).item()
            
            if top1 == self.vocab.word2idx["<EOS>"]:
                break
            
            response.append(top1)
            input = torch.tensor([[top1]], dtype=torch.long).to(self.device)
    
    # 转换为文本
    response_text = ''.join([self.vocab.idx2word[idx] for idx in response])
    return response_text
