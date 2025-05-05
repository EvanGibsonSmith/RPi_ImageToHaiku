from .camera import Camera
from .haiku_llama import HaikuLlama
from .vit import ViT
from .tts import TextToSpeech


class Ternary:
    def __init__(self, haiku_llama_path: str, vit_path: str, tts_path: str) -> None:
        self.camera = Camera()
        self.vit = ViT(vit_path)
        self.haiku_llama = HaikuLlama(haiku_llama_path)
        self.talker = TextToSpeech(tts_path)

    def __call__(self, verbose: bool = False) -> None:
        frame = self.camera()
        classes = self.vit(frame)
        if verbose:
            print(classes)
        text_out = self.haiku_llama(classes, 32, stop=["note"])
        if verbose:
            print(text_out)
        self.talker(text_out)
