from .camera import Camera
from .haiku_llama import HaikuLlama


class Ternary:
    def __init__(self) -> None:
        self.camera = Camera()
        self.haiku_llama = HaikuLlama()

        self.categories = [
            "nature",
            "love",
            "life",
            "death",
            "seasons",
            "time",
            "beauty",
            "tranquility",
            "harmony",
            "balance",
        ]

    def __call__(self) -> None:
        _ = self.camera()
        print(self.haiku_llama(self.categories))
