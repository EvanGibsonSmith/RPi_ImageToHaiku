import argparse
from ternary import Ternary


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="Ternary",
        description="A Ternary AI assistant for Raspberry Pi.",
    )
    parser.add_argument(
        "llama_path",
        type=str,
        default="models/llama-7b.gguf",
        help="Path to the Haiku Llama model.",
    )

    parser.add_argument(
       "vit_path",
       type=str,
    )

    parser.add_argument(
       "tts_path",
       type=str,
    )

    args = parser.parse_args()
    haiku_llama_path = args.llama_path
    vit_path = args.vit_path
    tts_path = args.tts_path
    if not haiku_llama_path:
        raise ValueError("Please provide a path to the Haiku Llama model.")

    ternary_pi = Ternary(haiku_llama_path, vit_path, tts_path)
    ternary_pi()
