"""
Скрипт для обучения модели генерации кода
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np

class CodeDataset(Dataset):
    def __init__(self, samples=500):
        self.samples = []
        patterns = [
            ("напиши функцию сложения двух чисел", "def add(a, b):\n    return a + b"),
            ("создай функцию факториала", "def factorial(n):\n    if n == 0:\n        return 1\n    else:\n        return n * factorial(n-1)"),
            ("функция для проверки числа на простоту", "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0:\n            return False\n    return True"),
        ]
        
        for desc, code in patterns:
            for i in range(samples // len(patterns)):
                self.samples.append({'description': desc, 'code': code})
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

class SimpleCodeModel(nn.Module):
    def __init__(self, vocab_size=5000, embed_dim=128, hidden_dim=256, num_layers=3):
        super(SimpleCodeModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        output = self.fc(lstm_out)
        return output

# Обучение модели
def train_model():
    print("🎯 Начало обучения модели...")
    
    model = SimpleCodeModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    dataset = CodeDataset(samples=300)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    losses = []
    for epoch in range(8):
        total_loss = 0
        for batch in dataloader:
            inputs = torch.randint(0, 5000, (16, 30))
            targets = torch.randint(0, 5000, (16, 30))
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, 5000), targets.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"📊 Эпоха {epoch+1}/8, Loss: {avg_loss:.4f}")
    
    # Сохраняем модель
    torch.save({
        'model_state_dict': model.state_dict(),
        'training_losses': losses,
        'final_loss': losses[-1],
        'model_config': {
            'vocab_size': 5000,
            'embed_dim': 128,
            'hidden_dim': 256,
            'num_layers': 3
        }
    }, 'model.pt')
    
    # Сохраняем график
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('График обучения модели')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('training_loss.png')
    
    print(f"✅ Модель сохранена! Финальный loss: {losses[-1]:.4f}")
    return losses

if __name__ == "__main__":
    losses = train_model()
