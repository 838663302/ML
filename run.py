import torch
import os
import config
from model import MyModel
from dataset import getLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def read_json():
    with open(config.DATASET_DIR / "vocab.txt", "r", encoding="utf-8") as f:
        vocab_list = [vocab.strip() for vocab in f.readlines()]
    word2id = {word: i for i, word in enumerate(vocab_list)}
    id2word = {i: word for i, word in enumerate(vocab_list)}
    return word2id, id2word

def train(word2id):
    model = MyModel(len(word2id)).to(device)
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    model.train()
    loss_value = float("inf")
    train_loader = getLoader(True)

    # 确保模型保存目录存在
    os.makedirs(config.MODEL_PATH.parent, exist_ok=True)
    writer = SummaryWriter(log_dir=f"logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    epoch_bar = tqdm(range(config.EPOCHS), desc="Epochs", unit="epoch")
    for epoch in epoch_bar:
        loss_avg = 0.0
        num_batches = 0
        batch_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}",
                         leave=False)
        for batch, target in batch_bar:
            batch = batch.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(batch)
            # output: (batch_size, vocab_size), target: (batch_size)
            loss = loss_func(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_avg += loss.item()
            num_batches += 1
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")
        avg_loss = loss_avg / num_batches if num_batches else 0.0
        scheduler.step()
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        epoch_bar.set_postfix(avg_loss=f"{avg_loss:.4f}",
                              lr=f"{optimizer.param_groups[0]['lr']:.6f}")
        if avg_loss < loss_value:
            loss_value = avg_loss
            torch.save(model.state_dict(), config.MODEL_PATH)
    writer.close()

def predict(input_str, word2id, id2word, model):
    tokens = [token.strip() for token in jieba.lcut(input_str)]
    indexedTokens = [word2id.get(token, 0) for token in tokens]
    model.eval()
    with torch.no_grad():
        input_ids = torch.tensor(indexedTokens, dtype=torch.long).unsqueeze(0).to(device)
        output = model(input_ids)
        topk_values, topk_indices = torch.topk(output, k=5, dim=1)
        topk_words = [id2word[idx.item()] for idx in topk_indices[0]]
        print(topk_words)
        print(topk_values)

    
if __name__ == "__main__":
    word2id, id2word = read_json()
    # model = MyModel(len(word2id)).to(device)
    # model.load_state_dict(torch.load(config.MODEL_PATH))
    
    train(word2id)