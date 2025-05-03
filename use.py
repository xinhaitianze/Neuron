from Neuron import *
import torch
import pickle
import jieba

def tokenize(text):
    return list(jieba.cut(text))

class SimpleChatbot:
    def __init__(self, model_path='best_model.pth', vocab_path='vocab.pkl'):
        # 加载词汇表
        with open(vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)
        
        # 初始化模型
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = Seq2SeqWithAttention(
            Encoder(len(self.vocab), 256, 2, 128),
            DecoderWithAttention(256, len(self.vocab), 2, 128)
        ).to(self.device)
        
        # 安全加载模型参数
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True),
            strict=True
        )
        self.model.eval()
    
    def generate_response(self, input_text):
        # 分词和转换为索引
        tokens = [self.vocab.word2idx.get(word, self.vocab.word2idx["<UNK>"]) for word in tokenize(input_text)]
        
        # 转换为张量
        input_tensor = torch.tensor([tokens], dtype=torch.long).to(self.device)
        
        # 创建虚拟目标张量，只包含 <SOS> 标记
        sos_token = torch.tensor([[self.vocab.word2idx["<SOS>"]]], dtype=torch.long).to(self.device)
        
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
                
                # 获取当前词的文本表示
                word = self.vocab.idx2word[top1]
                print(word, end='')  # 逐词输出
                
                # 更新输入
                input = torch.tensor([[top1]], dtype=torch.long).to(self.device)
        print()  # 换行

    def chat(self):
        print("开始对话（输入'exit'结束）")
        while True:
            user_input = input("You：")
            if user_input.lower() in ['退出', 'exit', 'quit']:
                break
            
            print("AI：", end='')  # 提示 AI 的回答
            self.generate_response(user_input)  # 调用 generate_response 方法

if __name__ == '__main__':
    bot = SimpleChatbot()
    bot.chat()