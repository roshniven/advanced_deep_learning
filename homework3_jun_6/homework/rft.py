from .base_llm import BaseLLM
from .sft import train_model as sft_train_model, test_model as sft_test_model
from pathlib import Path
from peft import PeftModel


def load() -> BaseLLM:
    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm

def train_model(
    output_dir: str = "homework/rft_model",

    r_lora: int = 32,
    lora_alpha: int = 128,
    num_train_epochs: int = 5,
    **kwargs,
):

    sft_train_model(
        output_dir=output_dir,
        use_rft_data=True,
        r_lora=r_lora,
        lora_alpha=lora_alpha,
        num_train_epochs=num_train_epochs,
        **kwargs,
    )


if __name__ == "__main__":
    from fire import Fire
    Fire({"train": train_model, "test": sft_test_model, "load": load})
