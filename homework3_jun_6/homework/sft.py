# homework/sft.py (Modified)
from .base_llm import BaseLLM
from .data import Dataset, benchmark
import torch
from transformers import TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from pathlib import Path
import json # Added import for json

# Make sure this is the path where your RFT data will be saved
RFT_DATASET_PATH = "data/rft.json" 

def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "sft_model" # Or "rft_model" if you want to explicitly name the RFT trained model
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
    # For RFT, the 'answer' here is the full CoT reasoning + <answer> tag
    full_text = f"{question} {answer}{tokenizer.eos_token}"

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    full = tokenizer(full_text, padding="max_length", truncation=True, max_length=128)

    input_ids = full["input_ids"]
    # Tokenize the question separately to get its length
    # It's crucial here that `question` is *just* the question, not the CoT prompt from CoTModel.
    # The `format_example` for RFT will handle this.
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
    Construct a question / answer pair for the original SFT task.
    This function will be reused for RFT, but the 'answer' for RFT will be the full
    Chain-of-Thought string, not just the float.
    """
    # This format_example is for the *original* SFT (question -> <answer>float</answer>)
    # For RFT, the answer will already be a string containing the CoT and <answer> tag.
    # We will need a specific format_fn for the RFT dataset.
    rounded_answer = f"<answer>{answer:.5f}</answer>"
    return {"question": prompt, "answer": rounded_answer}


def format_rft_example(question: str, true_answer: float, full_cot_text: str) -> dict[str, str]:
    """
    Construct a question / CoT answer pair for RFT.
    The 'answer' in this case is the full chain of thought string.
    """
    return {"question": question, "answer": full_cot_text}


class TokenizedDataset(torch.utils.data.Dataset):
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
        self.tokenized_data = []

        # Assuming data is either Dataset("train") (tuples of (question, answer_float))
        # or the RFT JSON data (tuples of (question, true_answer_float, full_cot_text))
        for item in self.data:
            if len(item) == 2: # Original SFT format (question, answer_float)
                formatted_data = self.format_fn(item[0], item[1])
            elif len(item) == 3: # RFT format (question, true_answer_float, full_cot_text)
                formatted_data = self.format_fn(item[0], item[1], item[2]) # Pass all args to format_rft_example
            else:
                raise ValueError(f"Unexpected data format: {item}")
            
            self.tokenized_data.append(tokenize(self.tokenizer, **formatted_data))


    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        return self.tokenized_data[idx]


def train_model(
    output_dir: str = "homework/sft_model", # Default output directory
    use_rft_data: bool = False, # New argument to switch between SFT and RFT data
    r_lora: int = 8, # LoRA rank, made configurable
    lora_alpha: int = 32, # LoRA alpha, made configurable
    num_train_epochs: int = 5, # Made configurable
    **kwargs,
):
    llm = BaseLLM()

    # Configure LoRA with configurable rank and alpha
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=r_lora,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        target_modules="all-linear",
        bias="none",
    )

    llm.model = get_peft_model(llm.model, peft_config)
    llm.model.print_trainable_parameters()

    if torch.cuda.is_available() and getattr(llm.model, "enable_input_require_grads", None):
        llm.model.enable_input_require_grads()
    
    # Override the format_prompt in BaseLLM for training purposes
    original_format_prompt = llm.format_prompt
    llm.format_prompt = lambda q: q # Simply return the question as is for training input

    # Determine which dataset to use based on use_rft_data
    if use_rft_data:
        print(f"Training with RFT dataset from {RFT_DATASET_PATH}")
        with open(RFT_DATASET_PATH, 'r') as f:
            rft_raw_data = json.load(f)
        train_dataset = TokenizedDataset(llm.tokenizer, rft_raw_data, format_rft_example)
        # For RFT, validation should still be on original data if we want to check numeric accuracy
        # Or you can create a RFT-style valid dataset too if you want to check CoT generation quality.
        # For this homework, let's keep original valid set for benchmarking.
        eval_dataset = TokenizedDataset(llm.tokenizer, Dataset("valid"), format_example)
    else:
        print("Training with standard SFT dataset")
        train_dataset = TokenizedDataset(llm.tokenizer, Dataset("train"), format_example)
        eval_dataset = TokenizedDataset(llm.tokenizer, Dataset("valid"), format_example)

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=3e-4,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        logging_dir=output_dir,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to="tensorboard",
        save_total_limit=1,
        push_to_hub=False,
        remove_unused_columns=False,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=llm.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

    trainer.save_model(output_dir)
    print(f"LoRA model saved to {output_dir}")

    llm.format_prompt = original_format_prompt # Restore original
    
    # For RFT, it's important to benchmark with the correct format_prompt (which is just the question)
    # The `test_model` function already handles this by setting llm.format_prompt = lambda q: q
    test_model(output_dir)


def test_model(ckpt_path: str):
    testset = Dataset("valid")
    llm = BaseLLM()

    from peft import PeftModel

    print(f"Loading LoRA model from {ckpt_path}")
    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)
    llm.model.eval()

    # For testing, we just provide the question and expect the model to generate CoT then answer
    # So the format_prompt should just be the question itself for a directly fine-tuned RFT model.
    llm.format_prompt = lambda q: q

    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})