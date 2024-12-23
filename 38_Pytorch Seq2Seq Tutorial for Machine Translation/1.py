import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import numpy as np
import random
import spacy
from torch.utils.tensorboard import SummaryWriter

spacy_eng = spacy.load("en_core_web_sm")
spacy_ger = spacy.load("de_core_web_sm")

def tokinizer_ger(text):
    return [tok.text for tok in spacy_ger.tokinizer(text)]

def tokinizer_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

german = Field(
    tokenize=tokinizer_ger,
    init_token="<sos>",
    eos_token="eos"
)

english = Field(
    tokenize=tokinizer_eng,
    init_token="<sos>",
    eos_token="<eos>"
)

train_data, validation_data, test_data = Multi30k(exts=('.de', '.en'), fields=(german, english))

german.build_vocab(train_data, max_size=1000, min_freq=2)
english.build_vocab(train_data, max_size=1000, min_freq=2)

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.hiddedn_size = hidden_size
        self.num_layers = num_layers
        
        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, num_layers, dropout=p)

    def forward(self, x):
        # x shape -> (seq_len, N)
        embedding = self.dropout(self.embedding(x))
        # embedding shape -> (seq_leng, N, embedding_size)
        output, (hidden, cell) = self.rnn(embedding)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, p):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p)

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, num_layers, dropout=p)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        # Shape of x -> N, but we want (1, N)
        x = x.unsqueeze(0)

        embedding = self.dropout(self.embedding(x))
        # Shape of embedding -> (1, N, embedding_size)

        output, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        # Shaphe of output -> (1, N, hidden_size)

        predictions = self.fc(output)
        # Shape of predictions -> (1, N, len_of_vocab)

        predictions.squeeze(0)

        return predictions, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder =  encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[0]
        target_len = target.shape[0]
        target_vocab_size = len(english.vocab)

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)

        hidden, cell = self.encoder(source)

        # Grab start token
        x = target[0]

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)

            outputs[t] = output

            best_guess = output.argmax(1)

            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs
    
# Save the model
def save_checkpoint(state, filename='my_checkpoint.pth.tar'):
    print("=> Saving checkpoint")
    torch.save(state, filename)

# Load the model
def load_checkpoint(checkpoint):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

# Training Hyperparameters
num_epochs = 32
learning_rate = 0.001
batch_size = 16

# Model Hyperparameters
load_model = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size_encoder = len(german.vocab)
input_size_decoder = len(english.vocab)
output_size = len(english.vocab)
encoder_embedding_size =  300
decoder_embedding_size = 300
hidden_size = 1024
num_layers = 2
enc_dropout = 0.5
dec_dropout = 0.5

# Tensorboard
writer = SummaryWriter(f'runs/loss_plot')
step = 0

train_iterator, valid_iterator, test_iterator  = BucketIterator(
    (train_data, validation_data, test_data),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key= lambda x: len(x.src),
    device=device
)

encoder_net = Encoder(
    input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout
).to(device)

decoder_net = Encoder(
    input_size_decoder, decoder_embedding_size, hidden_size, output_size, dec_dropout
).to(device)

model = Seq2Seq(encoder_net, decoder_net)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

pad_idx = english.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.ptar"), model, optimizer)

sentence = 'ein boot mit anderen mannern wird wird von einem groben pferdespann ans ufer gezogen'

for epoch in num_epochs:
    print(f"Epoch [{epoch} / {num_epochs}]")
    checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
    save_checkpoint(checkpoint)

    model.eval()
    
    translated_sentence = translated_sentence(model, sentence, german, english, device, max_length=50)
    print(f"Translated example sentance\n{translated_sentence}")

    for batch_idx, batch in enumerate(train_iterator):
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)

        output = model(inp_data, target)
        # output shape: (trg_len, batch_sizem output_dim)

        output = output[1:].reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()
        loss = criterion(output, target)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters())