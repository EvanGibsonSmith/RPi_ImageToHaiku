import torch.nn.utils.prune as prune
from tacotron2.loss_function import Tacotron2Loss
from speechbrain.inference.vocoders import HIFIGAN
import torchaudio
import os
import torch
from torch.utils.data import Dataset, DataLoader
from tacotron2.data_utils import TextMelLoader, TextMelCollate
import torch.nn as nn
from tqdm import tqdm
from tacotron2.text import text_to_sequence
import matplotlib.pyplot as plt

class ModifiedTacotron2Loss(nn.Module):
    def __init__(self):
        super(ModifiedTacotron2Loss, self).__init__()

    def forward(self, model_output, targets):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)
        
        mel_out, mel_out_postnet, gate_out, _ = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
            nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        return mel_loss + gate_loss
    
class MelDataset(Dataset):
    def __init__(self, data_dir):
        self.filepaths = sorted([
            os.path.join(data_dir, fname)
            for fname in os.listdir(data_dir)
            if fname.endswith(".pt")
        ])

    def __getitem__(self, idx):
        data = torch.load(self.filepaths[idx])
        return data["text"], data["text_len"], data["mel"], data["mel_len"]

    def __len__(self):
        return len(self.filepaths)

def get_pruned_tacotron2_value(prune_amount = 0.5, layer_idx=5, validation_size=None):
        # TODO run tacotron pruned on each layer and test against validation set of mel spectrograms collected
        nvidia_tacotron2 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tacotron2', model_math='fp32')
        nvidia_tacotron2 = nvidia_tacotron2.to('cuda')

        # TODO make function for specific layer and prune amount to run over values to graph

        # Check the number of pruned weights for each pruned layer
        # # # Prune tacotron
        for i, (name, module) in enumerate(nvidia_tacotron2.named_modules()): # TODO prune one layer at time for this analysis
            # Apply pruning to Conv1d or Conv2d layers
            if hasattr(module, 'weight'):
                print(f"TacoTron2: Pruning layer {name}")  # Confirm pruning layers
                prune.l1_unstructured(module, name='weight', amount=prune_amount)  # Prune 30% of the weights
                prune.remove(module, name='weight')  # Permanently remove the pruned weights

        pruned_weights_count = 0
        total_weights_count = 0
        for name, module in nvidia_tacotron2.named_modules():
            if hasattr(module, 'weight'):
                pruned_weights = module.weight == 0  # Identify pruned weights
                pruned_weights_count += pruned_weights.sum().item()
                total_weights_count += module.weight.numel()
        
        pruned_percentage = (pruned_weights_count / total_weights_count) * 100
        print(f"Pruned {pruned_weights_count} weights, {pruned_percentage:.2f}% of total weights in TacoTron2.")

        validation_files = r"tacotron2\get_dataset\ljs_audio_text_val_filelist.txt"

        # Add needed hparams from defaults in speechbrain pretrained model
        nvidia_tacotron2.hparams.text_cleaners = ['english_cleaners'] # Add text cleaner to load mel
        nvidia_tacotron2.hparams.max_wav_value=32768.0
        nvidia_tacotron2.hparams.sampling_rate=22050
        nvidia_tacotron2.hparams.filter_length=1024
        nvidia_tacotron2.hparams.hop_length=256
        nvidia_tacotron2.hparams.win_length=1024
        nvidia_tacotron2.hparams.n_mel_channels=80
        nvidia_tacotron2.hparams.mel_fmin=0.0
        nvidia_tacotron2.hparams.mel_fmax=8000.0
        nvidia_tacotron2.hparams.load_mel_from_disk = False
        nvidia_tacotron2.hparams.seed = 31415

        valset = TextMelLoader(validation_files, nvidia_tacotron2.hparams)

        val_sampler = None
        collate_fn = TextMelCollate(1)
        val_loader = DataLoader(valset, sampler=val_sampler, num_workers=1,
                                shuffle=False, batch_size=1,
                                pin_memory=False, collate_fn=collate_fn)

        total_loss = 0
        loss_func = Tacotron2Loss()

        for i, batch in tqdm(enumerate(val_loader)):
            if i==validation_size:
                break
            x, y = nvidia_tacotron2.parse_batch(batch)
            y_pred = nvidia_tacotron2(x)
            total_loss += loss_func(y_pred, y)

        # Collect total_loss for this portion of the graph
        if (validation_size!=None):
            length = validation_size
        else:
            length = len(val_loader)

        return nvidia_tacotron2, total_loss.item()/length

if __name__=="__main__":
    out = []
    #prune_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    prune_values = []
    for prune_value in tqdm(prune_values):
        out.append(get_pruned_tacotron2_value(prune_value, layer_idx=0, validation_size=20)[1])

    # Run tacotron
    prune_value = 0.9
    nvidia_tacotron2_pruned = get_pruned_tacotron2_value(0.9, validation_size=2)[0]

    # Example input sentence
    text = "Hello, how are you?"
    sequence = text_to_sequence(text, ['english_cleaners'])  # Adjust cleaner as needed
    sequence = torch.IntTensor(sequence)[None, :]  # (1, T)
    sequence = sequence.cuda()
    
    with torch.no_grad():
        mel_outputs, mel_outputs_postnet, gate_outputs, alignments = nvidia_tacotron2_pruned.inference(sequence)

    hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="text_to_speech/hifigan_model")
    hifigan_state_dict = hifi_gan.hparams.generator.state_dict()  # Extract state dictionary
    hifigan_model = hifi_gan.hparams.generator

    mel = mel_outputs_postnet  # or mel_outputs if needed
    mel = mel.float().cuda()
    waveforms = hifi_gan.decode_batch(mel)

    # Save the waveform
    torchaudio.save(f'test.wav', waveforms.squeeze(1), 22050)

    import numpy as np
    plt.plot(prune_values, list(-np.array(out)))
    plt.show()