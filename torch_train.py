import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import polars as pl
from sklearn.metrics import f1_score, accuracy_score
from collections import Counter

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_and_combine(file_path: str):
    """Load CSV and combine the three text columns with basic preprocessing."""
    df = pl.read_csv(file_path)
    
    # Combine keyword, location, and text
    df = df.with_columns(
        combined_text = (
            pl.col('keyword').fill_null('') + " " + 
            pl.col('location').fill_null('') + " " + 
            pl.col('text').fill_null('')
        )
    )
    
    # Basic preprocessing using Polars string methods
    df = df.with_columns(
        combined_text = pl.col('combined_text')
        .str.to_lowercase()
        .str.replace_all(r'http\S+|www\S+|https\S+', '') # Remove URLs
        .str.replace_all(r'\W', ' ') # Remove non-alphanumeric
        .str.replace_all(r'\s+', ' ') # Remove extra whitespace
        .str.strip_chars()
    )
    
    return df.select(['combined_text', 'target'])

class DisasterDataset(Dataset):
    def __init__(self, df: pl.DataFrame, vocab: dict[str,int], max_len: int = 50):
        self.texts = [self.tokenize(t, vocab, max_len) for t in df['combined_text']]
        self.labels = df['target'].to_list()
        
    def tokenize(self, text: str, vocab: dict[str,int], max_len: int):
        tokens = text.split()
        ids = [vocab.get(token, vocab['<UNK>']) for token in tokens]
        if len(ids) < max_len:
            ids += [vocab['<PAD>']] * (max_len - len(ids))
        else:
            ids = ids[:max_len]
        return ids

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.texts[idx], dtype=torch.long), 
            torch.tensor(self.labels[idx], dtype=torch.float)
        )

class DisasterClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, output_dim: int):
        super(DisasterClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True, num_layers=2, dropout=0.5)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, text: str):
        # text shape: [batch_size, max_len]
        embedded = self.embedding(text)
        # embedded shape: [batch_size, max_len, embedding_dim]
        _, (hidden, _) = self.lstm(embedded)
        
        # Concatenate the final forward and backward hidden states
        hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        out = self.fc(hidden_cat)
        return self.sigmoid(out)

def train():
    # Hyperparameters
    MAX_LEN = 64
    BATCH_SIZE = 32
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 128
    OUTPUT_DIM = 1
    LEARNING_RATE = 2e-4
    EPOCHS = 15

    print("Loading data...")
    train_df = load_and_combine('train_split.csv')
    test_df = load_and_combine('test_split.csv')

    # Build Vocabulary from training data
    print("Building vocabulary...")
    all_text = " ".join(train_df['combined_text'].to_list())
    words = all_text.split()
    word_counts = Counter(words)
    # Only include words that appear at least twice to reduce noise
    filtered_words = [word for word, count in word_counts.items() if count >= 2]
    vocab = {word: i + 2 for i, word in enumerate(filtered_words)}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    
    print(f"Vocabulary size: {len(vocab)}")

    train_dataset = DisasterDataset(train_df, vocab, MAX_LEN)
    test_dataset = DisasterDataset(test_df, vocab, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = DisasterClassifier(len(vocab), EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()

    print(f"Starting training on {device}...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            
            optimizer.zero_grad()
            predictions = model(texts).view(-1)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Evaluation
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for texts, labels in test_loader:
                texts, labels = texts.to(device), labels.to(device)
                predictions = model(texts).view(-1)
                preds = (predictions > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")

    print("Training complete.")

if __name__ == "__main__":
    train()
