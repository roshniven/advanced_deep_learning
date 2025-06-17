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
                "You are a helpful and concise assistant that performs unit conversions. "
                "Always show the correct conversion factor, multiply, and wrap the final result in <answer>...</answer>. "
                "Include a line of reasoning before the answer. Only return a number inside the answer tag."
            ),
        },
        {
            "role": "user",
            "content": "How much is 10 inches in centimeters?",
        },
        {
            "role": "assistant",
            "content": (
                "1 inch = 2.54 cm.\n"
                "10 inches * 2.54 = 25.4 cm.\n"
                "<answer>25.4</answer>"
            ),
        },
        {
            "role": "user",
            "content": "Convert 5 kilometers to miles.",
        },
        {
            "role": "assistant",
            "content": (
                "1 kilometer = 0.621371 miles.\n"
                "5 kilometers * 0.621371 = 3.106855 miles.\n"
                "<answer>3.106855</answer>"
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