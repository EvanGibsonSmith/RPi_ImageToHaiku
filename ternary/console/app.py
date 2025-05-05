import argparse
from ternary import Ternary


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="Ternary",
        description="A Ternary AI assistant for Raspberry Pi.",
    )
    parser.add_argument(
        "haiku-llama-path",
        type=str,
        default="models/llama-7b.gguf",
        help="Path to the Haiku Llama model.",
    )

    args = parser.parse_args()
    haiku_llama_path = args.haiku_llama_path
    if not haiku_llama_path:
        raise ValueError("Please provide a path to the Haiku Llama model.")

    ternary_pi = Ternary(haiku_llama_path)

    ternary_pi()
