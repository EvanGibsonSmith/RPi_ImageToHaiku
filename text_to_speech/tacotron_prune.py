from speechbrain.inference.TTS import Tacotron2
import torch.nn.utils.prune as prune
from tacotron2.loss_function import Tacotron2Loss

# TODO run tacotron pruned on each layer and test against validation set of mel spectrograms collected

tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="text_to_speech/tacotron2_model")
tacotron_state_dict = tacotron2.hparams.model.state_dict()  # Extract state dictionary
tacotron_model = tacotron2.hparams.model

prune_amount = 0.3

# TODO make function for specific layer and prune amount to run over values to graph

# Check the number of pruned weights for each pruned layer
# # # Prune tacotron
for i, (name, module) in enumerate(tacotron_model.named_modules()): # TODO prune one layer at time for this analysis

    if i == 10:
        # Apply pruning to Conv1d or Conv2d layers
        if hasattr(module, 'weight'):
            print(f"TacoTron2: Pruning layer {name}")  # Confirm pruning layers
            prune.l1_unstructured(module, name='weight', amount=prune_amount)  # Prune 30% of the weights
            prune.remove(module, name='weight')  # Permanently remove the pruned weights

pruned_weights_count = 0
total_weights_count = 0
for name, module in tacotron2.named_modules():
    if hasattr(module, 'weight'):
        pruned_weights = module.weight == 0  # Identify pruned weights
        pruned_weights_count += pruned_weights.sum().item()
        total_weights_count += module.weight.numel()

pruned_percentage = (pruned_weights_count / total_weights_count) * 100
print(f"Pruned {pruned_percentage:.2f}% of total weights in TacoTron2.")

dataloader = 
total_loss = 0
for sentence, target_mel_spec in dataloader:
    mel_output, mel_length, alignment = tacotron2([sentence])
    total_loss += Tacotron2Loss(mel_output, target_mel_spec)

# Collect total_loss for this portion of the graph