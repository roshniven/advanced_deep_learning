# homework/rft.py
from .base_llm import BaseLLM
# Import train_model and test_model directly from sft.py for reuse
from .sft import train_model as sft_train_model, test_model as sft_test_model 


def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "rft_model" # This is the expected output directory for RFT model
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def train_model(
    output_dir: str = "homework/rft_model", # Default output directory for RFT model
    r_lora: int = 16, # Increased rank for RFT as suggested in prompt
    lora_alpha: int = 64, # Increased alpha
    num_train_epochs: int = 5, # Standard epochs
    **kwargs,
):
    # Reuse the train_model logic from sft.py, but force use_rft_data=True
    # and provide RFT-specific default parameters for LoRA and output directory.
    sft_train_model(
        output_dir=output_dir,
        use_rft_data=True, # Crucially set to True for RFT
        r_lora=r_lora,
        lora_alpha=lora_alpha,
        num_train_epochs=num_train_epochs,
        **kwargs,
    )


if __name__ == "__main__":
    from fire import Fire

    # Use the test_model from sft.py directly, as it handles loading and benchmarking
    # The 'load' command will be for the rft_model specifically
    Fire({"train": train_model, "test": sft_test_model, "load": load})