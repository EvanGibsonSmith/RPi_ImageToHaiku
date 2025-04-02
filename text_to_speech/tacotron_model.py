import torch
import torch.nn as nn
import torch.nn.functional as F
import speechbrain

class ConvNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, bias=True):
        super(ConvNorm, self).__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

    def forward(self, x):
        return self.conv(x)

class LinearNorm(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        return self.linear_layer(x)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.convolutions = nn.ModuleList([
            nn.Sequential(
                ConvNorm(512, 512, kernel_size=5),
                nn.BatchNorm1d(512)
            ) for _ in range(3)
        ])
        self.lstm = nn.LSTM(512, 256, batch_first=True, bidirectional=True)
    
    def forward(self, x):
        for conv in self.convolutions:
            x = F.relu(conv(x))
        x, _ = self.lstm(x.transpose(1, 2))
        return x

class Prenet(nn.Module):
    def __init__(self):
        super(Prenet, self).__init__()
        self.layers = nn.ModuleList([
            LinearNorm(80, 256, bias=False),
            LinearNorm(256, 256, bias=False)
        ])
    
    def forward(self, x):
        for linear in self.layers:
            x = F.relu(linear(x))
        return x

class Postnet(nn.Module):
    def __init__(self):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList([
            nn.Sequential(
                ConvNorm(80, 512, kernel_size=5),
                nn.BatchNorm1d(512)
            )
        ] + [
            nn.Sequential(
                ConvNorm(512, 512, kernel_size=5),
                nn.BatchNorm1d(512)
            ) for _ in range(3)
        ] + [
            nn.Sequential(
                ConvNorm(512, 80, kernel_size=5),
                nn.BatchNorm1d(80)
            )
        ])
    
    def forward(self, x):
        for conv in self.convolutions:
            x = F.relu(conv(x))
        return x

class MyTacotron2(nn.Module):
    def __init__(self):
        super(MyTacotron2, self).__init__()
        self.embedding = nn.Embedding(148, 512)
        self.encoder = Encoder()
        self.prenet = Prenet()
        self.decoder_rnn = nn.LSTMCell(1536, 1024)
        self.linear_projection = LinearNorm(1536, 80, bias=True)
        self.gate_layer = LinearNorm(1536, 1, bias=True)
        self.postnet = Postnet()
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.prenet(x)
        x, _ = self.decoder_rnn(x)
        x = self.linear_projection(x)
        x = self.postnet(x)
        return x

    def text_to_seq(self, txt):
        """Encodes raw text into a tensor with a customer text-to-sequence function"""
        sequence = self.hparams.text_to_sequence(txt, self.text_cleaners)
        return sequence, len(sequence)

    def encode_batch(self, texts):
        """Computes mel-spectrogram for a list of texts

        Texts must be sorted in decreasing order on their lengths

        Arguments
        ---------
        texts: List[str]
            texts to be encoded into spectrogram

        Returns
        -------
        tensors of output spectrograms, output lengths and alignments
        """
        with torch.no_grad():
            inputs = [
                {
                    "text_sequences": torch.tensor(
                        self.text_to_seq(item)[0], device=self.device
                    )
                }
                for item in texts
            ]
            inputs = speechbrain.dataio.batch.PaddedBatch(inputs)

            lens = [self.text_to_seq(item)[1] for item in texts]
            assert lens == sorted(
                lens, reverse=True
            ), "input lengths must be sorted in decreasing order"
            input_lengths = torch.tensor(lens, device=self.device)

            mel_outputs_postnet, mel_lengths, alignments = self.forward(
                inputs.text_sequences.data, input_lengths
            )
        return mel_outputs_postnet, mel_lengths, alignments

    def encode_text(self, text):
        """Runs inference for a single text str"""
        return self.encode_batch([text])