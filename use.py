import os
os.system("cls")
print("Loading now, wait a moment...")
from Neuron import *
import torch
import pickle
import jieba
import re
from rich import print
from rich.panel import Panel
from rich.progress import Progress
from rich.console import Console
from rich.markdown import Markdown
from rich.style import Style
from rich.text import Text
import time
from typing import Optional

# 初始化富文本控制台
console = Console()

def tokenize(text):
    return list(jieba.cut(text))

class SimpleChatbot:
    def __init__(self, model_path='best_model.pth', vocab_path='vocab.pkl'):
        # 初始化生成参数
        self.generation_config = {
            'temperature': 0.7,
            'top_k': 50,
            'top_p': 0.9,
            'max_length': 400
        }
        
        # 添加加载动画
        with Progress(transient=True) as progress:
            task = progress.add_task("[cyan]正在初始化Neuron...", total=100)
            
            # 加载词汇表
            for _ in range(30):
                time.sleep(0.02)
                progress.update(task, advance=1)
            with open(vocab_path, 'rb') as f:
                self.vocab = pickle.load(f)
                
            # 模型加载进度
            for _ in range(70):
                time.sleep(0.01)
                progress.update(task, advance=1)
            
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

    def _apply_temperature(self, logits: torch.Tensor) -> torch.Tensor:
        """应用温度参数到logits"""
        temperature = self.generation_config['temperature']
        if temperature > 0:
            return logits / temperature
        return logits

    def _apply_top_k_p(self, logits: torch.Tensor) -> torch.Tensor:
        """应用top-k和top-p过滤"""
        # Top-k过滤
        if self.generation_config['top_k'] > 0:
            indices_to_remove = logits < torch.topk(logits, self.generation_config['top_k'])[0][..., -1, None]
            logits[indices_to_remove] = -float('Inf')
        
        # Top-p过滤
        if 0 < self.generation_config['top_p'] < 1:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # 移除累积概率超过top_p的token
            sorted_indices_to_remove = cumulative_probs > self.generation_config['top_p']
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, 
                index=sorted_indices, 
                src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = -float('Inf')
        
        return logits

    def generate_response(self, input_text: str) -> Optional[str]:
        try:
            # 输入净化
            input_text = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fff\s]', '', input_text).lower()
            
            # 分词处理
            tokens = [self.vocab.word2idx.get(word, self.vocab.word2idx["<UNK>"]) 
                     for word in tokenize(input_text)]
            
            # 转换为张量
            input_tensor = torch.tensor([tokens], dtype=torch.long).to(self.device)
            sos_token = torch.tensor([[self.vocab.word2idx["<SOS>"]]], dtype=torch.long).to(self.device)
            
            # 编码器处理
            encoder_out, hidden = self.model.encoder(input_tensor)
            input = sos_token
            
            # 生成响应带打字机效果
            response = []
            with console.status("[bold green]思考中...", spinner="dots"):
                with torch.no_grad():
                    for _ in range(self.generation_config['max_length']):
                        output, hidden, _ = self.model.decoder(input, hidden, encoder_out)
                        
                        # 应用生成参数
                        logits = output[0, -1]
                        logits = self._apply_temperature(logits)
                        logits = self._apply_top_k_p(logits)
                        
                        # 采样
                        probs = torch.softmax(logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1).item()
                        
                        if next_token == self.vocab.word2idx["<EOS>"]:
                            break
                        
                        word = self.vocab.idx2word[next_token]
                        response.append(word)
                        input = torch.tensor([[next_token]], dtype=torch.long).to(self.device)
            
            print("Neuron:")
            print(Panel(
                "".join(response),
                subtitle=f"[italic]NeuronAI V1.1",
                style=Style(color="cornflower_blue", bgcolor="black"),
                border_style="dark_blue"
            ))
            return "".join(response)
            
        except Exception as e:
            error_text = Text(f"系统异常: {str(e)}", style="bold red on black")
            print(Panel(error_text, title="⚠️ 错误"))
            return None

    def _show_help(self):
        """显示帮助信息"""
        help_text = """
        [bold]可用命令：[/bold]
        /temp [数值]  - 设置温度 (0.1~2.0)
        /topk [数值]  - 设置Top-K采样 (0关闭)
        /topp [数值]  - 设置Top-P采样 (0~1)
        /reset        - 重置所有参数
        /help         - 显示本帮助
        /exit         - 退出程序
        
        [yellow]当前参数设置：[/yellow]
        """ + "\n".join([f"{k}: {v}" for k, v in self.generation_config.items()])
        
        print(Panel.fit(
            Markdown(help_text),
            title="帮助信息",
            style="dim white"
        ))

    def chat(self):
        # 欢迎界面
        console.print(Panel(
            "[bold green]Neuron 对话系统已激活[/]\n"
            "[yellow]输入 /help 查看可用命令[/]",
            title="NeuronAI 已启动",
        ))
        
        while True:
            try:
                # 美化输入提示
                user_input = console.input("\n[bold green]👤 您[/]: ")
                
                # 命令处理
                if user_input.startswith('/'):
                    cmd = user_input[1:].split()
                    if not cmd:
                        continue
                        
                    if cmd[0] == 'temp' and len(cmd) > 1:
                        try:
                            new_temp = float(cmd[1])
                            if 0.1 <= new_temp <= 2.0:
                                self.generation_config['temperature'] = new_temp
                                console.print(f"[green]温度已设置为 {new_temp}[/]")
                            else:
                                console.print("[red]温度值必须在0.1到2.0之间[/]")
                        except ValueError:
                            console.print("[red]无效的温度值[/]")
                    
                    elif cmd[0] == 'topk' and len(cmd) > 1:
                        try:
                            new_topk = int(cmd[1])
                            if new_topk >= 0:
                                self.generation_config['top_k'] = new_topk
                                console.print(f"[green]Top-K已设置为 {new_topk}[/]")
                            else:
                                console.print("[red]Top-K值必须≥0[/]")
                        except ValueError:
                            console.print("[red]无效的Top-K值[/]")
                    
                    elif cmd[0] == 'topp' and len(cmd) > 1:
                        try:
                            new_topp = float(cmd[1])
                            if 0 <= new_topp <= 1:
                                self.generation_config['top_p'] = new_topp
                                console.print(f"[green]Top-P已设置为 {new_topp}[/]")
                            else:
                                console.print("[red]Top-P值必须在0到1之间[/]")
                        except ValueError:
                            console.print("[red]无效的Top-P值[/]")
                    
                    elif cmd[0] == 'reset':
                        self.generation_config = {
                            'temperature': 0.7,
                            'top_k': 50,
                            'top_p': 0.9,
                            'max_length': 50
                        }
                        console.print("[green]参数已重置为默认值[/]")
                    
                    elif cmd[0] == 'help':
                        self._show_help()
                    
                    elif cmd[0] in ['exit', 'quit']:
                        console.print(Panel(
                            "[italic yellow]正在关闭程序...",
                            title="系统状态",
                            border_style="yellow"
                        ))
                        time.sleep(1)
                        os.system("cls")
                        break
                    
                    else:
                        console.print("[red]未知命令，输入/help查看可用命令[/]")
                    continue
                
                self.generate_response(user_input.lower())

            except KeyboardInterrupt:
                console.print("\n[red bold]⚠ 安全中断已触发[/]")
                break

if __name__ == '__main__':
    bot = SimpleChatbot()
    
    # 版本信息
    panel = Panel(
    "欢迎使用 NeuronAI 1.1\n"
    "Visit [link=https://xinhaitianze.github.io/Neuron.html]https://xinhaitianze.github.io/Neuron.html[/link] for more information.\n"
    "输入 /exit 退出程序",
    title="欢迎使用 NeuronAI 1.1",
    subtitle="你可以输入 /help 查看控制命令"
    )

    console.print(panel)
    bot.chat()