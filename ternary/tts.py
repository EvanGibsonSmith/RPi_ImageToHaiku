from speechbrain.inference.vocoders import HIFIGAN
from speechbrain.inference.TTS import Tacotron2
import sounddevice as sd
import numpy as np
import torch
import numpy as np
from torchvision import datasets, transforms
#from haiku_llama import HaikuLlama
from vit import ViT

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
        mel_output, mel_length, alignment = self.tacotron2([sentence])

        waveforms = self.hifi_gan.decode_batch(mel_output)
        out = waveforms.squeeze(0).squeeze(0)

        sd.play(out, 22050)
        sd.wait()
        return out

if __name__=="__main__":
    vit = ViT(path="ternary/model.tflite")
    #haiku_llama = HaikuLlama()
    tts = TextToSpeech(tacotron_state_dict_path="d30e20.pth")
    tts("Doctor Livingston I assume, clever to say.")

    # Load dummy CIFAR-100
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to shape (C, H, W), range [0.0, 1.0]
    ])

    # Load the test set
    img_idx = 0
    cifar100_test = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    # Load the CIFAR-100 dataset
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()

    # Select a real image (for example, the first test image)
    real_image = np.expand_dims(x_test[0], axis=0).astype(np.float32)  # Shape (1, 32, 32, 3)

    # Now pass the real image to your model
    output_data = vit(real_image)
    print("Predictions for real image:", output_data)

    # Get one sample
    image_tensor, label = cifar100_test[img_idx]  # image_tensor is shape (3, 32, 32)
    image_tensor = image_tensor.numpy()

    classes = vit(image_tensor)
    print(classes)
    #haiku = haiku_llama(classes)
    

