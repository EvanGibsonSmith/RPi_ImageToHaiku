from speechbrain.inference.vocoders import HIFIGAN
from speechbrain.inference.TTS import Tacotron2
from scipy.io.wavfile import write
import numpy as np
import torch
import numpy as np
from torchvision import datasets, transforms
#from haiku_llama import HaikuLlama
import subprocess
import os

class TextToSpeech:

    def __init__(self, tacotron_state_dict_path):
        self.hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="text_to_speech/hifigan_model")
        self.hifigan_state_dict = self.hifi_gan.hparams.generator.state_dict()  # Extract state dictionary
        self.hifigan_model = self.hifi_gan.hparams.generator

        self.tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="text_to_speech/tacotron2_model")
        self.tacotron_state_dict = self.tacotron2.hparams.model.state_dict()  # Extract state dictionary
        self.tacotron_model = self.tacotron2.hparams.model

        # Ensure that the loaded state dict matches the model
        self.tacotron_model = self.tacotron2.hparams.model

        # Load the custom state dict
        tacotron_state_dict = torch.load(tacotron_state_dict_path)
        self.tacotron_model.load_state_dict(tacotron_state_dict)

    def __call__(self, sentence):
        # Run tacotron
        print(sentence)
        mel_output, mel_length, alignment = self.tacotron2([sentence])

        waveforms = self.hifi_gan.decode_batch(mel_output)
        out = waveforms.squeeze().cpu().detach().numpy()
        print(out)
        audio_array = out / np.max(np.abs(out))
        audio_int16 = (out*32767).astype(np.int16)
        print(audio_int16)

        write("temp.wav", 22050, audio_int16)

        try:
             subprocess.run(["aplay", "temp.wav"], check=True)
        except FileNotFoundError:
             print("error")

        return out

if __name__=="__main__":
    tts = TextToSpeech(tacotron_state_dict_path="d30e20.pth")
    tts("Doctor Livingston I assume, clever to say.")


