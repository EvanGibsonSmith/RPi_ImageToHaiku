from .camera import Camera
from .haiku_llama import HaikuLlama
from .vit import ViT

class Ternary:
    def __init__(self) -> None:
        self.camera = Camera()
        self.vit = ViT()
        # self.haiku_llama = HaikuLlama()

        self.categories = [
            "truck",
            "beaver",
            "airplane",
            "cat",
            "raspberry pi"
        ]

    def __call__(self) -> None:
        frame = self.camera()
        classes = self.vit(frame)
        print(classes)
	# print(self.haiku_llama(self.categories))
