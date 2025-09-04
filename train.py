from Neuron import *
import pickle
import os
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"GPU是否可用：{torch.cuda.is_available()}")
def validate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for question_batch, answer_batch in dataloader:
            question_batch = question_batch.to(device)
            answer_batch = answer_batch.to(device)
            
            output = model(question_batch, answer_batch)
            
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = answer_batch[:, 1:].contiguous().view(-1)
            
            loss = criterion(output, trg)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def read_data(data_dir='data'):
    """加载data目录下所有.AiTrainData文件"""
    questions = []
    answers = []
    data_files = [f for f in os.listdir(data_dir) if f.endswith('.AiTrainData')]
    
    if not data_files:
        raise FileNotFoundError(f"在 {data_dir} 目录下未找到任何.AiTrainData文件")
    
    print(f"找到 {len(data_files)} 个训练文件:")
    for filename in tqdm(data_files, desc='加载文件中'):
        with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as f:
            current_q, current_a = [], []
            for line in f:
                line = line.strip()
                if line.startswith('？'):
                    if current_q:
                        questions.append('\n'.join(current_q))
                        answers.append('\n'.join(current_a))
                        current_q, current_a = [], []
                    current_q.append(line[1:])
                elif line.startswith('！'):
                    current_a.append(line[1:])
            
            if current_q:
                questions.append('\n'.join(current_q))
                answers.append('\n'.join(current_a))
    
    print(f"合并后总数据量: {len(questions)} 组问答对")
    return questions, answers

if __name__ == "__main__":
    #读取训练文件
    questions, answers = read_data()
    print(f"成功加载 {len(questions)} 组问答对")
    
    #创建词汇表
    vocab = Vocabulary(min_freq=1)
    vocab.build_vocab(questions + answers)

    #important！！！！！！！！！！
    #保存词汇表
    with open('vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    print(f"词汇表已保存，大小: {len(vocab)}")

    #创建数据集和数据加载器
    dataset = ChatbotDataset(questions, answers, vocab)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    #初始化模型
    input_size = len(vocab)
    output_size = len(vocab)
    hidden_size = 256
    num_layers = 4
    embedding_dim = 128

    encoder = Encoder(input_size, hidden_size, num_layers, embedding_dim)
    decoder = DecoderWithAttention(hidden_size, output_size, num_layers, embedding_dim)
    model = Seq2SeqWithAttention(encoder, decoder).to(device)

    #定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=0)  #忽略填充的 <PAD> 索引
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    #训练模型
    num_epochs = 200
    patience = 20  #如果验证损失在连续5个轮次中没有下降，则停止训练
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for question_batch, answer_batch in dataloader:
            question_batch = question_batch.to(device)
            answer_batch = answer_batch.to(device)
            
            optimizer.zero_grad()
            output = model(question_batch, answer_batch)
            
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = answer_batch[:, 1:].contiguous().view(-1)
            
            loss = criterion(output, trg)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}')

        #验证损失
        val_loss = validate(model, dataloader, criterion)
        print(f'Validation Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        elif val_loss <= 0.100:
            break
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping triggered")
                break

        #保存模型
        torch.save(model.state_dict(), 'best_model.pth')


#再见了Torch框架，我要去尝试ChatterBot了，拜拜 :)
#2025.5.11

#tm的 ChatterBot 简直是依托答辩，还是Torch好
#2025.5.14