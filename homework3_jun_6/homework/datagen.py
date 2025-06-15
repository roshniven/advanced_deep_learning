import json
from pathlib import Path
from tqdm import tqdm
from .cot import CoTModel
from .data import Dataset, benchmark

def generate_dataset(
    output_path: str = "data/rft.json",
    num_samples_per_question: int = 20,
    temperature: float = 0.7,
    max_questions: int = -1,
):
    """
    Generates a dataset for RFT by using a CoTModel to produce multiple completions
    for each question, then selecting the one with the correct answer.
    """
    cot_model = CoTModel()

    train_dataset = Dataset("train")
    rft_data = []

    print(f"Generating RFT dataset with {num_samples_per_question} samples per question...")

    for i, (question, true_answer) in enumerate(tqdm(train_dataset, desc="Generating RFT data")):
        if max_questions != -1 and i >= max_questions:
            break

        prompt = cot_model.format_prompt(question)    
        generations = cot_model.batched_generate(
            [prompt],
            num_return_sequences=num_samples_per_question,
            temperature=temperature
        )

        if not generations:
            continue

        best_rollout = None
        for generated_text in generations[0]:
            parsed_answer = cot_model.parse_answer(generated_text)

            if abs(parsed_answer - true_answer) < 1e-4:
                rft_data.append([question, true_answer, generated_text.strip()])
                best_rollout = generated_text
                break

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
