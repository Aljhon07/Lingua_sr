class CNNLayerNorm(nn.Module):
   def __init__(self, n_feats):
       super(CNNLayerNorm, self).__init__()
       self.layer_norm = nn.LayerNorm(n_feats)

   def forward(self, x):
       # x (batch, channel, feature, time)
       x = x.transpose(2, 3).contiguous() # (batch, channel, time, feature)
       x = self.layer_norm(x)
       return x.transpose(2, 3).contiguous() # (batch, channel, feature, time) 
   

class ResidualCNN(nn.Module):
   def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
       super(ResidualCNN, self).__init__()

       self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel//2)
       self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel//2)
       self.dropout1 = nn.Dropout(dropout)
       self.dropout2 = nn.Dropout(dropout)
       self.layer_norm1 = CNNLayerNorm(n_feats)
       self.layer_norm2 = CNNLayerNorm(n_feats)

   def forward(self, x):
       residual = x  # (batch, channel, feature, time)
       x = self.layer_norm1(x)
       x = F.gelu(x)        
       x = self.dropout1(x)
       x = self.cnn1(x)
       x = self.layer_norm2(x)
       x = F.gelu(x)
       x = self.dropout2(x)
       x = self.cnn2(x)
       x += residual
       return x # (batch, channel, feature, time)
       
class BidirectionalGRU(nn.Module):

   def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
       super(BidirectionalGRU, self).__init__()

       self.BiGRU = nn.GRU(
           input_size=rnn_dim, hidden_size=hidden_size,
           num_layers=1, batch_first=batch_first, bidirectional=True)
       self.layer_norm = nn.LayerNorm(rnn_dim)
       self.dropout = nn.Dropout(dropout)

   def forward(self, x):
       x = self.layer_norm(x)
       x = F.gelu(x)
       x, _ = self.BiGRU(x)
       x = self.dropout(x)
       return x


class SpeechRecognitionModel(nn.Module):

   def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_feats, stride=2, dropout=0.1):
       super(SpeechRecognitionModel, self).__init__()
       n_feats = n_feats//2
       self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3//2)  

       self.rescnn_layers = nn.Sequential(*[
           ResidualCNN(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats)
           for _ in range(n_cnn_layers)
       ])
       self.fully_connected = nn.Linear(n_feats*32, rnn_dim)
       self.birnn_layers = nn.Sequential(*[
           BidirectionalGRU(rnn_dim=rnn_dim if i==0 else rnn_dim*2,
                            hidden_size=rnn_dim, dropout=dropout, batch_first=i==0)
           for i in range(n_rnn_layers)
       ])
       self.classifier = nn.Sequential(
           nn.Linear(rnn_dim*2, rnn_dim), 
           nn.GELU(),
           nn.Dropout(dropout),
           nn.Linear(rnn_dim, n_class)
       )

   def forward(self, x):
       x = self.cnn(x)
       x = self.rescnn_layers(x)
       sizes = x.size()
       x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
       x = x.transpose(1, 2) # (batch, time, feature)
       x = self.fully_connected(x)
       x = self.birnn_layers(x)
       x = self.classifier(x)
       return x
   

def train(model, device, train_loader, criterion, optimizer, scheduler, epoch, iter_meter, experiment):
   model.train()
   data_len = len(train_loader.dataset)
    for batch_idx, _data in enumerate(train_loader):
        spectrograms, labels, input_lengths, label_lengths = _data
        spectrograms, labels = spectrograms.to(device), labels.to(device)

        optimizer.zero_grad()

        output = model(spectrograms)  # (batch, time, n_class)
        output = F.log_softmax(output, dim=2)
        output = output.transpose(0, 1) # (time, batch, n_class)

        loss = criterion(output, labels, input_lengths, label_lengths)
        loss.backward()

        optimizer.step()
        scheduler.step()
        if batch_idx % 100 == 0 or batch_idx == data_len:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(spectrograms), data_len,
                100. * batch_idx / len(train_loader), loss.item()))

def main(learning_rate=5e-4, batch_size=20, epochs=10,
    train_url="train-clean-100", test_url="test-clean",
    experiment=Experiment(api_key='dummy_key', disabled=True)):

    hparams = {
       "n_cnn_layers": 3,
       "n_rnn_layers": 5,
       "rnn_dim": 512,
       "n_class": 29,
       "n_feats": 128,
       "stride": 2,
       "dropout": 0.1,
       "learning_rate": learning_rate,
       "batch_size": batch_size,
       "epochs": epochs
   }


   use_cuda = torch.cuda.is_available()
   torch.manual_seed(7)
   device = torch.device("cuda" if use_cuda else "cpu")

   if not os.path.isdir("./data"):
       os.makedirs("./data")

   train_dataset = torchaudio.datasets.LIBRISPEECH("./data", url=train_url, download=True)
   test_dataset = torchaudio.datasets.LIBRISPEECH("./data", url=test_url, download=True)

   kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
   train_loader = data.DataLoader(dataset=train_dataset,
                               batch_size=hparams['batch_size'],
                               shuffle=True,
                               collate_fn=lambda x: data_processing(x, 'train'),
                               **kwargs)
   test_loader = data.DataLoader(dataset=test_dataset,
                               batch_size=hparams['batch_size'],
                               shuffle=False,
                               collate_fn=lambda x: data_processing(x, 'valid'),
                               **kwargs)

   model = SpeechRecognitionModel(
       hparams['n_cnn_layers'], hparams['n_rnn_layers'], hparams['rnn_dim'],
       hparams['n_class'], hparams['n_feats'], hparams['stride'], hparams['dropout']
       ).to(device)

   print(model)
   print('Num Model Parameters', sum([param.nelement() for param in model.parameters()]))

   optimizer = optim.AdamW(model.parameters(), hparams['learning_rate'])
   criterion = nn.CTCLoss(blank=28).to(device)
   scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hparams['learning_rate'],
                                           steps_per_epoch=int(len(train_loader)),
                                           epochs=hparams['epochs'],
                                           anneal_strategy='linear')

   iter_meter = IterMeter()
   for epoch in range(1, epochs + 1):
       train(model, device, train_loader, criterion, optimizer, scheduler, epoch, iter_meter, experiment)
       test(model, device, test_loader, criterion, epoch, iter_meter, experiment)
       
       

import torch
import torch.nn as nn
import torchaudio

char_map_str = """
' 0
1 a
2 b
3 c
4 d
5 e
6 f
7 g
8 h
9 i
10 j
11 k
12 l
13 m
14 n
15 o
16 p
17 q
18 r
19 s
20 t
21 u
22 v
23 w
24 x
25 y
26 z
27
"""

class TextTransform:
    """Maps characters to integers and vice versa"""
    def __init__(self):
        char_map_str = char_map_str
        self.char_map = {}
        self.index_map = {}
        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch
        self.index_map[1] = ' '

    def text_to_int(self, text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = self.char_map['']
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence

    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return ''.join(string).replace('', ' ')

train_audio_transforms = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
    torchaudio.transforms.TimeMasking(time_mask_param=35)
)

valid_audio_transforms = torchaudio.transforms.MelSpectrogram()
text_transform = TextTransform()

def data_processing(data, data_type="train"):
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    for (waveform, _, utterance, _, _, _) in data:
        if data_type == 'train':
            spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        else:
            spec = valid_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        spectrograms.append(spec)
        label = torch.Tensor(text_transform.text_to_int(utterance.lower()))
        labels.append(label)
        input_lengths.append(spec.shape[0] // 2)
        label_lengths.append(len(label))

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
    return spectrograms, labels, input_lengths, label_lengths