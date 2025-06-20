from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def __init__(self):
        super().__init__()
    
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """

        messages = [
        {
            "role": "system",
            "content": (
                "You are a math problem solver specialist. You apply logical reasoning to a problem and solve it using a clear, step-by-step approach."
                "Wrap the final result in <answer>...</answer>. Be concise, precise, and accurate."
            ),
        },
        {
            "role": "user",
            "content": "What is the conversion from mph to m/s for 7 mph?",
        },
        {
            "role": "assistant",
            "content": (
                "1 mph = 0.44704 m/s."
                "7 mph * 0.44704 = 3.12928 m/s.\n"
                "<answer>3.12928</answer>"
            ),
        },
        {
            "role": "user",
            "content": question.strip(),
        },
    ]

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})