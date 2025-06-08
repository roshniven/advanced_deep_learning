# homework/datagen.py
import json
from pathlib import Path
from tqdm import tqdm
from .cot import CoTModel # Import CoTModel
from .data import Dataset, benchmark

def generate_dataset(
    output_path: str = "data/rft.json",
    num_samples_per_question: int = 20, # Generate 20 completions per question
    temperature: float = 0.7, # Use a non-zero temperature for diverse outputs
    max_questions: int = -1, # Set to a specific number to limit dataset size during testing
):
    """
    Generates a dataset for RFT by using a CoTModel to produce multiple completions
    for each question, then selecting the one with the correct answer.
    """
    cot_model = CoTModel()
    
    # Temporarily change the checkpoint of BaseLLM to the 1.7B Instruct model for better CoT generation
    # This change should be done carefully or make CoTModel load a specific checkpoint
    # For simplicity, we'll assume CoTModel in cot.py is already loading the desired model
    # (e.g., HuggingFaceTB/SmolLM2-1.7B-Instruct if specified in its __init__)
    # If CoTModel loads the default BaseLLM checkpoint, you might need to adjust it there
    # or pass a specific checkpoint to its constructor if it supports it.
    # For now, let's assume CoTModel is set up to use the 1.7B Instruct or similar for good CoT.

    train_dataset = Dataset("train")
    rft_data = []
    
    print(f"Generating RFT dataset with {num_samples_per_question} samples per question...")

    for i, (question, true_answer) in enumerate(tqdm(train_dataset, desc="Generating RFT data")):
        if max_questions != -1 and i >= max_questions:
            break

        # Generate multiple diverse completions using batched_generate
        # The prompt should be formatted by CoTModel's format_prompt
        prompt = cot_model.format_prompt(question)
        
        # batched_generate returns list[list[str]] when num_return_sequences is set
        generations = cot_model.batched_generate(
            [prompt], 
            num_return_sequences=num_samples_per_question, 
            temperature=temperature
        )
        
        # generations will be a list containing one sublist of num_samples_per_question strings
        # because we passed only one prompt.
        if not generations:
            continue
        
        best_rollout = None
        
        for generated_text in generations[0]: # Iterate through the generated options for the single prompt
            parsed_answer = cot_model.parse_answer(generated_text)
            
            # Check if the parsed answer is correct (allowing for float comparison tolerance)
            # Use a small tolerance for float comparison
            if abs(parsed_answer - true_answer) < 1e-4:
                # Found a correct answer, now we need to extract the "reasoning" part
                # The generated_text includes the prompt, so we need to remove it
                # The actual "reasoning" starts after the last user turn.
                # Since format_prompt adds add_generation_prompt=True, the model directly
                # generates the assistant's response.
                
                # We want to save: [original_question, true_answer, full_generated_response_with_reasoning]
                rft_data.append([question, true_answer, generated_text.strip()])
                best_rollout = generated_text # Mark that we found a valid rollout
                break # Move to the next question as we found a correct one

    # Save the generated RFT dataset to a JSON file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(rft_data, f, indent=2)
    
    print(f"Generated RFT dataset with {len(rft_data)} entries saved to {output_path}")


def test_datagen():
    # A simple test to run the datagen process
    # Generate a small dataset for testing purposes
    print("Running test_datagen...")
    generate_dataset(output_path="data/rft_test.json", max_questions=10)
    print("Test dataset generation complete. Check data/rft_test.json")


if __name__ == "__main__":
    from fire import Fire
    Fire({"generate": generate_dataset, "test": test_datagen})