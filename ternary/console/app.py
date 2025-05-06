import RPi.GPIO as GPIO
import argparse
from ternary import Ternary
import subprocess

PIN = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # internal pull‑up


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="Ternary",
        description="A Ternary AI assistant for Raspberry Pi.",
    )
    parser.add_argument(
        "--llama_path",
        type=str,
        default="/home/arnavsacheti/Documents/Ternary-Pi/models/lora-model.q4_0.gguf",
        help="Path to the Haiku Llama model.",
    )

    parser.add_argument(
        "--vit_path",
        type=str,
        default="/home/arnavsacheti/Documents/Ternary-Pi/models/model.tflite",
        help="Path to the Vision Transformer model.",
    )

    parser.add_argument(
        "--tts_path",
        type=str,
        default="/home/arnavsacheti/Documents/Ternary-Pi/models/d30e20.pth",
        help="Path to the Text-to-Speech model.",
    )

    args = parser.parse_args()
    haiku_llama_path = args.llama_path
    vit_path = args.vit_path
    tts_path = args.tts_path
    if not haiku_llama_path:
        raise ValueError("Please provide a path to the Haiku Llama model.")

    ternary_pi = Ternary(haiku_llama_path, vit_path, tts_path)

    try:
        while True:
            try:
                print("Waiting for button press…")
                subprocess.run(
                    [
                        "aplay",
                        "/home/arnavsacheti/Documents/Ternary-Pi/sound_effects/wait_for_button.wav",
                    ],
                    check=True,
                )
            except FileNotFoundError:
                print(
                    "ALSA audio player not found. Please install it or use a different audio player."
                )
            GPIO.wait_for_edge(PIN, GPIO.FALLING)  # blocks until LOW edge

            try:
                print("Button pressed!")
                subprocess.run(
                    [
                        "aplay",
                        "/home/arnavsacheti/Documents/Ternary-Pi/sound_effects/beep.wav",
                    ],
                    check=True,
                )
            except FileNotFoundError:
                print(
                    "ALSA audio player not found. Please install it or use a different audio player."
                )

            ternary_pi(verbose=True)
    except KeyboardInterrupt:
        print("Exiting…")
    finally:
        print("Cleaning up GPIO…")
        GPIO.cleanup()
