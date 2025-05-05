from .haiku_llama import HaikuLlama


class Ternary:
    def __init__(self) -> None:
        self.haiku_llama = HaikuLlama("arnavsacheti/autotrain-llama-haiku")
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

    def __run__(self) -> None:
        print(self.haiku_llama(self.categories))

    def __call__(self) -> None:
        self.__run__()
