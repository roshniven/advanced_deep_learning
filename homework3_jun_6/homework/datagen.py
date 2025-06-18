import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from .cot import CoTModel
from .data import Dataset, benchmark
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_dataset(
    output_path: str = "data/rft.json",
    num_samples_per_question: int = 10,
    temperature: float = 0.7,
    max_questions: int = -1,
):
    """
    Generates a dataset for RFT by using a CoTModel to produce multiple completions
    for each question, then selecting the one with the correct answer.
    """
    cot_model = CoTModel()
    instruct_model_checkpoint = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    cot_model.tokenizer = AutoTokenizer.from_pretrained(instruct_model_checkpoint)
    cot_model.model = AutoModelForCausalLM.from_pretrained(instruct_model_checkpoint).to(cot_model.device)

    train_dataset = Dataset("train")
    rft_data = []

    for i, (question, true_answer) in enumerate(tqdm(train_dataset, desc="Generating RFT data")):
        if max_questions != -1 and i >= max_questions:
            break

        prompt = cot_model.format_prompt(question)    
        generations = cot_model.batched_generate(
            [prompt],
            num_return_sequences=num_samples_per_question,
            temperature=temperature
        ) [0]

        if not generations:
            continue

        best_rollout = None
        min_error = float('inf')

        for generated_text in generations:
            parsed_answer = cot_model.parse_answer(generated_text)
            current_error = abs(parsed_answer - true_answer)/abs(true_answer)

            if not np.isnan(parsed_answer) and current_error < 1e-1:                
                if current_error < min_error:
                    min_error = current_error
                    best_rollout = generated_text

        # After checking all generations, if we found a valid answer, add the best one found.
        if best_rollout is not None:
            rft_data.append([question, true_answer, best_rollout.strip()])

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(rft_data, f, indent=2)

    print(f"Generated RFT dataset with {len(rft_data)} entries saved to {output_path}")

def test_datagen():
    print("Running test_datagen...")
    generate_dataset(output_path="data/rft_test.json", max_questions=10)
    print("Test dataset generation complete. Check data/rft_test.json")


if __name__ == "__main__":
    from fire import Fire
    Fire({"generate": generate_dataset, "test": test_datagen})
