import os
from llama_cpp import Llama  # pip install llama-cpp-python


class HaikuLlama:
    """
    Tiny helper around llama-cpp-python.

    Example
    -------
    >>> from haiku_llama import HaikuLlama
    >>> llm = HaikuLlama("models/lora-model.q4_0.gguf")
    >>> print(llm("Write a haiku about pruning weights.", max_tokens=60))
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_threads: int | None = None,
        **llama_kwargs,
    ):
        """
        Parameters
        ----------
        model_path : str
            Path to the *.gguf* file you produced with `quantize`.
        n_ctx : int, default 4096
            Context window (tokens). 4096 is safe for 7‑B models on a Pi‑5 or desktop.
        n_threads : int | None
            CPU threads to use. None → `os.cpu_count()`.
        llama_kwargs : dict
            Any extra kwargs accepted by `llama_cpp.Llama`
            (e.g. n_gpu_layers, rope_freq_base).
        """
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads or os.cpu_count(),
            **llama_kwargs,
        )

    def __call__(
        self,
        categories: list[str],
        max_tokens: int = 256,
        temperature: float = 0.7,
        stop: list[str] | None = None,
        **gen_kwargs,
    ) -> str:
        """
        Run a single prompt and return the generated text.

        Parameters
        ----------
        prompt : str
            The prompt to feed the model (plain text).
        max_tokens : int, default 256
            Maximum new tokens to generate.
        temperature : float, default 0.7
            Sampling temperature (0 → deterministic).
        stop : list[str] | None
            Optional stop strings.
        gen_kwargs : dict
            Any additional generation kwargs accepted by `llm(...)`
            (top_p, repeat_penalty, etc.).

        Returns
        -------
        str
            The model’s reply (text only).
        """
        out = self.llm(
            f"Can you compose a singular haiku about these {len(categories)} categories: \"{', '.join(categories)}\"."
            " This is being fed to a TTS model, so it should be short and sweet."
            " Please do not include any extra text.",
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            **gen_kwargs,
        )
        # llama-cpp-python returns a dict with 'choices'
        return out["choices"][0]["text"].lstrip()
