from .camera import Camera
from .haiku_llama import HaikuLlama
from .vit import ViT


class Ternary:
    def __init__(self, haiku_llama_path: str, vit_path: str) -> None:
        self.camera = Camera()
        self.vit = ViT(vit_path)
        self.haiku_llama = HaikuLlama(haiku_llama_path)

    def __call__(self) -> None:
        frame = self.camera()
        classes = self.vit(frame)
        print(self.haiku_llama(classes))
