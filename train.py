from Neuron import *
import pickle
import re

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

def read_data(file_path):
    questions = []
    answers = []
    current_question = []
    current_answer = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
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

def preprocess_question(question):
    # 将所有字母转换为小写
    question = question.lower()
    # 使用正则表达式去除标点符号和其他非字母字符
    question = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fff\s]', '', question)
    return question
if __name__ == "__main__":
    # 读取训练文件
    questions, answers = read_data('data/Q&A.AiTrainData')
    print(f"成功加载 {len(questions)} 组问答对")
    
    # 创建词汇表
    vocab = Vocabulary(min_freq=1)
    vocab.build_vocab(questions + answers)

    # important！！！！！！！！！！
    # 保存词汇表
    with open('vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    print(f"词汇表已保存，大小: {len(vocab)}")

    # 创建数据集和数据加载器
    dataset = ChatbotDataset(questions, answers, vocab)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    # 初始化模型
    input_size = len(vocab)
    output_size = len(vocab)
    hidden_size = 256
    num_layers = 2
    embedding_dim = 128

    encoder = Encoder(input_size, hidden_size, num_layers, embedding_dim)
    decoder = DecoderWithAttention(hidden_size, output_size, num_layers, embedding_dim)
    model = Seq2SeqWithAttention(encoder, decoder).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略填充的 <PAD> 索引
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 150
    patience = 20  # 如果验证损失在连续5个轮次中没有下降，则停止训练
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

        # 验证损失
        val_loss = validate(model, dataloader, criterion)
        print(f'Validation Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping triggered")
                break

    # 保存模型
    torch.save(model.state_dict(), 'best_model.pth')


