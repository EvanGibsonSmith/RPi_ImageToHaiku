from speechbrain.inference.TTS import Tacotron2
from tacotron2.model import Tacotron2 as NVIDIATacotron2
import torch.nn.utils.prune as prune
from tacotron2.loss_function import Tacotron2Loss
from speechbrain.inference.vocoders import HIFIGAN
import tacotron2
import torchaudio
import os
import torch
from torch.utils.data import Dataset, DataLoader
from tacotron2.data_utils import TextMelLoader, TextMelCollate
import torch.nn as nn
from tqdm import tqdm
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

def get_pruned_tacotron2_value(prune_amount = 0.5, validation_size=None):
        # TODO run tacotron pruned on each layer and test against validation set of mel spectrograms collected

        tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="text_to_speech/tacotron2_model")
        tacotron_state_dict = tacotron2.hparams.model.state_dict()  # Extract state dictionary
        tacotron_model = tacotron2.hparams.model

        tacotron2.hparams.fp16_run = False 
        nvidia_tacotron2 = NVIDIATacotron2(hparams=tacotron2.hparams).to('cuda')

        # TODO make function for specific layer and prune amount to run over values to graph

        # Check the number of pruned weights for each pruned layer
        # # # Prune tacotron
        for i, (name, module) in enumerate(nvidia_tacotron2.named_modules()): # TODO prune one layer at time for this analysis

            #if i == 10:
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
        print(f"Pruned {pruned_percentage:.2f}% of total weights in TacoTron2.")

        validation_files = r"tacotron2\get_dataset\ljs_audio_text_val_filelist.txt"

        # Add needed hparams from defaults in speechbrain pretrained model
        tacotron2.hparams.text_cleaners = ['english_cleaners'] # Add text cleaner to load mel
        tacotron2.hparams.max_wav_value=32768.0
        tacotron2.hparams.sampling_rate=22050
        tacotron2.hparams.filter_length=1024
        tacotron2.hparams.hop_length=256
        tacotron2.hparams.win_length=1024
        tacotron2.hparams.n_mel_channels=80
        tacotron2.hparams.mel_fmin=0.0
        tacotron2.hparams.mel_fmax=8000.0
        tacotron2.hparams.load_mel_from_disk = False
        tacotron2.hparams.seed = 31415


        valset = TextMelLoader(validation_files, tacotron2.hparams)

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
    prune_values = [0.9]
    #for prune_value in tqdm(prune_values):
    #    out.append(get_pruned_tacotron2_value(prune_value, validation_size=20)[1])


    # Run tacotron
    prune_value = 0.9
    nvidia_tacotron2_pruned = get_pruned_tacotron2_value(0.9, validation_size=2)[0]

    plt.plot(prune_values, out)
    plt.show()