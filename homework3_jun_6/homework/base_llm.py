from typing import overload
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class BaseLLM:
    def __init__(self, checkpoint=checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
        self.device = device

    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into an input to SmolLM2. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        You don't need to change this function for now.
        """
        return question

    def parse_answer(self, answer: str) -> float:
        """
        Parse the <answer></answer> tag and return a float.
        This function is somewhat robust to output errors (e.g. missing </answer> tags).
        """
        try:
            return float(answer.split("<answer>")[1].split("</answer>")[0])
        except (IndexError, ValueError):
            return float("nan")

    def generate(self, prompt: str) -> str:
        """
        (Optional) Implement this method first and then implement batched_generate below.
        It is much easier to implement generation without batching.

        The overall flow is the same:
        - tokenize the prompt with self.tokenizer
        - call self.model.generate
        - decode the outputs with self.tokenizer.decode

        """
        return self.batched_generate([prompt])[0]

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: None = None, temperature: float = 0
    ) -> list[str]:
        """
        Batched version of `generate` method.
        This version returns a single generation for each prompt.
        """

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: int, temperature: float = 0
    ) -> list[list[str]]:
        """
        Batched version of `generate` method.
        This version returns a list of generation for each prompt.
        """

    def batched_generate(self, prompts: list[str], num_return_sequences: int | None = None, temperature: float = 0
) -> list[str] | list[list[str]]:
    # Micro-batching for memory management
      micro_batch_size = 32
      if len(prompts) > micro_batch_size:
          return [
              r
              for idx in tqdm(
                  range(0, len(prompts), micro_batch_size),
                  desc=f"LLM Running on Micro Batches (size={micro_batch_size})",
              )
              for r in self.batched_generate(
                  prompts[idx : idx + micro_batch_size], num_return_sequences, temperature
              )
          ]

      # Set tokenizer padding direction
      self.tokenizer.padding_side = "left"

      # Tokenize with padding
      inputs = self.tokenizer(
          prompts,
          return_tensors="pt",
          padding=True,
          truncation=True,
      ).to(self.device)

      # Prepare generation arguments
      gen_kwargs = {
        "max_new_tokens": 50,
        "eos_token_id": self.tokenizer.eos_token_id,
        "do_sample": temperature > 0,
        **({"temperature": temperature} if temperature > 0 else {}),
      }

      if num_return_sequences is not None:
          gen_kwargs["num_return_sequences"] = num_return_sequences

      # Generate outputs
      outputs = self.model.generate(
          input_ids=inputs["input_ids"],
          attention_mask=inputs["attention_mask"],
          **gen_kwargs,
      )

      # Slice off the prompt tokens to only keep generated tokens
      num_input_tokens = inputs["input_ids"].shape[1]
      generated_tokens = outputs[:, num_input_tokens:]

      # Decode to strings
      decoded = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

      # Return result
      if num_return_sequences is None:
          return decoded  # flat list of strings
      else:
          # group flat list into nested list: one sublist per prompt
          return [
              decoded[i * num_return_sequences : (i + 1) * num_return_sequences]
              for i in range(len(prompts))
          ]

    def answer(self, *questions) -> list[float]:
          """
          Answer questions given as individual string arguments.
          """
          # Convert each question
          prompts = [self.format_prompt(q) for q in questions]
          generations = self.batched_generate(prompts)
          return [self.parse_answer(g) for g in generations]


def test_model():
    # The following code simply tests of the BaseLLM is able to complete text.
    # It should produce garbage answers, but it should not crash.
    # In my case it talks about cats eating cats, and dogs being happy.
    testset = ["The cat went up", "The dog went down"]
    model = BaseLLM()
    for t in testset:
        print("testing generate function")
        print("input", t)
        answer = model.generate(t)
        print("output", answer)
    answers = model.batched_generate(testset)
    print(answers)


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model})