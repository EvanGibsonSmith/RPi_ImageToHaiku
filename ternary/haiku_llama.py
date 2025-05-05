from typing import Literal
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel


class HaikuLlama:

    def __init__(
        self,
        adapter_repo: str = "arnavsacheti/autotrain-llama-haiku",
        base_model_repo: str = "meta-llama/Llama-3.1-8B-Instruct",
    ) -> None:
        self.adapter_repo = adapter_repo
        self.base_model_repo = base_model_repo

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_repo)
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_repo, device_map=self.device, torch_dtype="auto"
        )
        self.model = PeftModel.from_pretrained(
            base_model, self.adapter_repo, device_map=self.device, torch_dtype="auto"
        ).to(self.device)
        self.model.eval()

    @property
    def device(self) -> Literal["cuda", "mps", "cpu"]:
        # if torch.cuda.is_available():
        #     return "cuda"
        # elif torch.backends.mps.is_available():
        #     return "mps"
        # else:
        return "cpu"

    def __call__(self, categories: list[str]) -> str:
        """
        Generate a haiku based on the provided categories.

        Args:
            categories (list[str]): A list of categories to base the haiku on.

        Returns:
            str: The generated haiku.
        """
        if not categories:
            raise ValueError("Categories list cannot be empty.")
        message = [
            {
                "role": "system",
                "content": "You are a poet specialising in creating Haiku. \nYour haiku consist of three lines, with five syllables in the first line, seven in the second, and five in the third.\nBeyond being technically correct, your haiku should also be beautiful and meaningful. \n Output ONLY one Haiku.",
            },
            {
                "role": "user",
                "content": f"Can you compose a singular haiku about these {len(categories)} categories: \"{', '.join(categories)}",
            },
        ]

        input_ids = self.tokenizer.apply_chat_template(
            conversation=message,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device)

        output_ids = self.model.generate(input_ids, max_new_tokens=256)
        response = self.tokenizer.decode(
            output_ids[0][input_ids.shape[1] :], skip_special_tokens=True
        )

        return response
