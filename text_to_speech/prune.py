from generator import HifiganGenerator
import speechbrain
from speechbrain.inference.vocoders import HIFIGAN
from speechbrain.inference.TTS import Tacotron2
from speechbrain.lobes.models.Tacotron2 import Tacotron2 as Manual_Tacotron2
from speechbrain.utils.text_to_sequence import text_to_sequence
#from tacotron_model import MyTacotron2 TODO remove this when working other way
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.optim as optim
import torchaudio

hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="text_to_speech/hifigan_model")
hifigan_state_dict = hifi_gan.hparams.generator.state_dict()  # Extract state dictionary
hifigan_model = hifi_gan.hparams.generator

tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="text_to_speech/tacotron2_model")
tacotron_state_dict = tacotron2.hparams.model.state_dict()  # Extract state dictionary
tacotron_model = tacotron2.hparams.model

hifi_prune_amount = 0.9
tacotron2_prune_amount = 0.3
# # # Prune hifi gan
# TODO not working for hifigan pruning
for name, module in hifigan_model.named_modules():
    if hasattr(module, 'weight'):
        print(f"HiFi: Pruning layer {name}")  # Confirm pruning layers
        module.weight = nn.Parameter(module.weight)
        prune.l1_unstructured(module, name='weight', amount=hifi_prune_amount)  # Prune 30% of the weights
        prune.remove(module, name='weight')  # Permanently remove the pruned weights

# Check the number of pruned weights for each pruned layer
for name, module in hifigan_model.named_modules():
    if hasattr(module, 'weight'):
        pruned_weights = module.weight == 0  # Identify pruned weights (zeroed out)
        num_pruned = pruned_weights.sum().item()  # Count the pruned weights
        total_weights = module.weight.numel()  # Total number of weights in the layer
        print(f"Layer {name}: Pruned {num_pruned} out of {total_weights} weights ({(num_pruned / total_weights) * 100:.2f}% pruned)")

# # # Prune tacotron
for name, module in tacotron_model.named_modules():
    # Apply pruning to Conv1d or Conv2d layers
    if hasattr(module, 'weight'):
        print(f"TacoTron2: Pruning layer {name}")  # Confirm pruning layers
        prune.l1_unstructured(module, name='weight', amount=tacotron2_prune_amount)  # Prune 30% of the weights
        prune.remove(module, name='weight')  # Permanently remove the pruned weights

# Check the number of pruned weights for each pruned layer
for name, module in tacotron_model.named_modules():
    if hasattr(module, 'weight'):
        pruned_weights = module.weight == 0  # Identify pruned weights (zeroed out)
        num_pruned = pruned_weights.sum().item()  # Count the pruned weights
        total_weights = module.weight.numel()  # Total number of weights in the layer
        print(f"Layer {name}: Pruned {num_pruned} out of {total_weights} weights ({(num_pruned / total_weights) * 100:.2f}% pruned)")

# HiFi GAN Prune
# TODO note this is pruning the discriminator which is not needed at all
#hifi_gan_pruned_state_dict = {}
#for name, module in pruned_hifi_gan.named_modules():
#    if hasattr(module, 'weight'):
#        # Apply pruning mask if it exists
#        if hasattr(module, 'weight_mask'):
#            # Set the weights to zero wherever the mask is zero
#            module.weight.data.mul_(module.weight_mask)  # Zero out pruned weights using the mask
#        
#        # Store pruned weights in the state dict
#        hifi_gan_pruned_state_dict[name + ".weight"] = module.weight
        
# Check if pruned weights are indeed affecting model performance
pruned_weights_count = 0
total_weights_count = 0
for name, module in hifi_gan.named_modules():
    if hasattr(module, 'weight'):
        pruned_weights = module.weight == 0  # Identify pruned weights
        pruned_weights_count += pruned_weights.sum().item()
        total_weights_count += module.weight.numel()

pruned_percentage = (pruned_weights_count / total_weights_count) * 100
print(f"Pruned {pruned_percentage:.2f}% of weights in HiFi-GAN.")

pruned_weights_count = 0
total_weights_count = 0
for name, module in tacotron2.named_modules():
    if hasattr(module, 'weight'):
        pruned_weights = module.weight == 0  # Identify pruned weights
        pruned_weights_count += pruned_weights.sum().item()
        total_weights_count += module.weight.numel()

pruned_percentage = (pruned_weights_count / total_weights_count) * 100
print(f"Pruned {pruned_percentage:.2f}% of weights in TacoTron2.")

# Run tacotron
sentence = "Roses are red, violets are blue, sugar is sweet, and so are you"
mel_output, mel_length, alignment = tacotron2([sentence])

# TODO make working with this instead MANUAL TACOTRON
#tacotron_manual = Manual_Tacotron2

# Running Vocoder (spectrogram-to-waveform)
waveforms = hifi_gan.decode_batch(mel_output)

# Save the waveform
torchaudio.save(f'text_to_speech/audios/hifi_prune_{hifi_prune_amount}_tacotron2_prune_{tacotron2_prune_amount}_said_{sentence}.wav', waveforms.squeeze(1), 22050)