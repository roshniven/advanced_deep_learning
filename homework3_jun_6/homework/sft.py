from .base_llm import BaseLLM
from .data import Dataset, benchmark
import torch
from transformers import TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from pathlib import Path


def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "sft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def tokenize(tokenizer, question: str, answer: str):
    """
    Tokenize a data element.
    We first append the <EOS> token to the question / answer pair.
    Then we tokenize and construct the ground truth `labels`.
    `labels[i] == -100` for the question or masked out parts, since we only want to supervise
    the answer.
    """
    full_text = f"{question} {answer}{tokenizer.eos_token}"

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    full = tokenizer(full_text, padding="max_length", truncation=True, max_length=128)

    input_ids = full["input_ids"]
    # Tokenize the question separately to get its length
    question_len = len(tokenizer(question)["input_ids"])

    # Create labels: mask out the prompt part
    labels = [-100] * question_len + input_ids[question_len:]

    # Ensure labels are -100 for padded tokens
    for i in range(len(labels)):
        if full["attention_mask"][i] == 0:
            labels[i] = -100

    full["labels"] = labels
    return full


def format_example(prompt: str, answer: float) -> dict[str, str]:
    """
    Construct a question / answer pair. Consider rounding the answer to make it easier for the LLM.
    """
    # Round the answer to a reasonable number of decimal places for consistency and LLM performance
    # For unit conversions, 3-5 decimal places are usually sufficient
    rounded_answer = f"<answer>{answer:.5f}</answer>"
    return {"question": prompt, "answer": rounded_answer}


class TokenizedDataset(torch.utils.data.Dataset): # Inherit from torch.utils.data.Dataset
    def __init__(self, tokenizer, data: Dataset, format_fn):
        """
        Use the
        - BaseLLM.tokenizer
        - Dataset
        - format_fn which converts a data element into a dict with entries
          - question: str
          - answer: str
        """
        self.format_fn = format_fn
        self.tokenizer = tokenizer
        self.data = data
        # Pre-tokenize all data during initialization for efficiency
        self.tokenized_data = []
        for question, answer in self.data:
            formatted_data = self.format_fn(question, answer)
            self.tokenized_data.append(tokenize(self.tokenizer, **formatted_data))


    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        return self.tokenized_data[idx]


def train_model(
    output_dir: str = "homework/sft_model", # Default output directory
    **kwargs,
):
    llm = BaseLLM()

    # Configure LoRA
    # You might need to adjust r and lora_alpha based on model size constraints.
    # A common rule of thumb is lora_alpha = 2 * r or 4 * r.
    # For a 360M model, a rank of 8-16 might be reasonable to keep adapter size small.
    # Let's try r=8, lora_alpha=32
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32, # Typically lora_alpha is a multiple of r
        lora_dropout=0.1,
        target_modules="all-linear",
        bias="none",
    )

    llm.model = get_peft_model(llm.model, peft_config)
    llm.model.print_trainable_parameters() # This will show the number of trainable LoRA parameters

    if torch.cuda.is_available() and getattr(llm.model, "enable_input_require_grads", None):
        llm.model.enable_input_require_grads()
    
    # Override the format_prompt in BaseLLM for SFT
    # In SFT, we don't use chat templates but directly ask for completion.
    original_format_prompt = llm.format_prompt
    llm.format_prompt = lambda q: q # Simply return the question as is for SFT input

    train_dataset = TokenizedDataset(llm.tokenizer, Dataset("train"), format_example)
    eval_dataset = TokenizedDataset(llm.tokenizer, Dataset("valid"), format_example) # Use valid set for evaluation

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=3e-4, # A common learning rate for LoRA
        num_train_epochs=1,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=1, # Adjust if you have smaller GPU memory
        gradient_checkpointing=True, # Saves GPU memory
        logging_dir=output_dir,
        logging_steps=100,
        eval_strategy="epoch", # Evaluate at the end of each epoch
        save_strategy="epoch",       # Save checkpoint at the end of each epoch
        report_to="tensorboard",
        save_total_limit=1, # Only keep the last checkpoint
        push_to_hub=False,
        remove_unused_columns=False, # Important for custom datasets
        load_best_model_at_end=True, # Load the best model based on eval_loss
    )

    trainer = Trainer(
        model=llm.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

    # Save the final LoRA adapter
    trainer.save_model(output_dir)
    print(f"LoRA model saved to {output_dir}")

    # Restore original format_prompt for potential subsequent calls if needed
    llm.format_prompt = original_format_prompt
    
    test_model(output_dir)


def test_model(ckpt_path: str):
    testset = Dataset("valid")
    llm = BaseLLM()

    # Load the model with LoRA adapters
    from peft import PeftModel

    print(f"Loading LoRA model from {ckpt_path}")
    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)
    llm.model.eval() # Set model to evaluation mode

    # Override the format_prompt for evaluation (same as during SFT training)
    llm.format_prompt = lambda q: q

    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
