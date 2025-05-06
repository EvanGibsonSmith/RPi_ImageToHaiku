import argparse
from ternary import Ternary
import subprocess

# Check if the script is running on a Raspberry Pi
import os

ON_RPI = os.uname().machine == "armv7l"
if ON_RPI:
    import RPi.GPIO as GPIO

    PIN = 17
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # internal pull‑up
    ternary_pi_dir = "~/Documents/Ternary-Pi/"
    play_cmd = "aplay"
else:
    play_cmd = "open"
    ternary_pi_dir = "./"


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="Ternary",
        description="A Ternary AI assistant for Raspberry Pi.",
    )
    parser.add_argument(
        "--llama_path",
        type=str,
        default=f"{ternary_pi_dir}models/lora-model.q4_0.gguf",
        help="Path to the Haiku Llama model.",
    )

    parser.add_argument(
        "--vit_path",
        type=str,
        default=f"{ternary_pi_dir}models/model.tflite",
        help="Path to the Vision Transformer model.",
    )

    parser.add_argument(
        "--tts_path",
        type=str,
        default=f"{ternary_pi_dir}models/d30e20.pth",
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
                        play_cmd,
                        f"{ternary_pi_dir}sound_effects/wait_for_button.wav",
                    ],
                    check=True,
                )
            except FileNotFoundError:
                print(
                    "ALSA audio player not found. Please install it or use a different audio player."
                )
            if ON_RPI:
                GPIO.wait_for_edge(PIN, GPIO.FALLING)  # blocks until LOW edge
            else:
                input("Press Enter to simulate button press…")

            try:
                print("Button pressed!")
                subprocess.run(
                    [
                        play_cmd,
                        f"{ternary_pi_dir}sound_effects/beep.wav",
                    ],
                    check=True,
                )
            except FileNotFoundError:
                print(
                    "ALSA audio player not found. Please install it or use a different audio player."
                )

            ternary_pi(verbose=True, play_cmd=play_cmd)
    except KeyboardInterrupt:
        print("Exiting…")
    finally:
        if ON_RPI:
            print("Cleaning up GPIO…")
            GPIO.cleanup()
