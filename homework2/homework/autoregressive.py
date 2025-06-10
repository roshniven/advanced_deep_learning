import abc

import torch
import torch.nn as nn
from torch.nn import functional as F


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "AutoregressiveModel"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    # Make sure to map to the correct device when loading
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.load(model_path, map_location=device, weights_only=False)


class Autoregressive(abc.ABC):
    """
    Base class for all autoregressive models.
    Implement a specific model below.
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Take a tensor x (B, h, w) if integers as input.
        Produce a probability over the next token as an output (B, h, w, n_token).
        Make sure the model is auto-regressive:
          - The first output result[:, 0, 0] does not depend on any input
          - The second output result[:, 0, 1] depends only on x[:, 0, 0]
          - etc.

        Hint 1: Flatten the tensor into a sequence.
        Hint 2: A positional embedding can help, but is not required.
        Hint 3: You need to shift the input sequence by 1 position. Do this after embedding the
                values, and before passing them through your model. (torch.concat or
                torch.nn.ConstantPad1d both work)
        """

    def generate(self, B: int = 1, h: int = 20, w: int = 30, device=None) -> torch.Tensor:  # noqa
        """
        Use your generative model to produce B new token images of size (B, h, w) and type (int/long).
        """


class AutoregressiveModel(torch.nn.Module, Autoregressive):
    """
    An implementation of an auto-regressive model using a Transformer architecture.
    The input is a batch of tokenized images (integers), and the output is a
    probability distribution over the next token for each position in the image.

    Key components:
    - nn.Embedding: To convert integer tokens to dense vectors.
    - Positional Embedding: To provide the model with sequence order information.
    - nn.TransformerEncoder: To process the sequence of embeddings.
    - Causal Mask: To ensure the model is auto-regressive and doesn't see future tokens.
    """

    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10, n_head: int = 4, n_layers: int = 4, max_seq_len: int = 20 * 30):
        super().__init__()
        self.d_latent = d_latent
        self.n_tokens = n_tokens
        
        # 1. Token Embedding: Maps each integer token to a d_latent dimensional vector
        # We now map n_tokens + 1 for an implicit "start" token, but no explicit parameter.
        self.token_embedding = nn.Embedding(n_tokens + 1, d_latent) 
        
        # 2. Positional Embedding: Adds positional information to the token embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_latent))

        # 3. Transformer Encoder: A stack of transformer layers
        # batch_first=True is crucial for using (B, Seq, Dim) tensor shapes
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_latent, nhead=n_head, dim_feedforward=d_latent * 4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 4. Output Layer: Projects the transformer output back to the token vocabulary size
        self.output_head = nn.Linear(d_latent, n_tokens)
        
        # Removed: self.start_token = nn.Parameter(torch.randn(1, 1, d_latent))

    def _generate_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """
        Generates a square causal mask for the transformer.
        The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        return nn.Transformer.generate_square_subsequent_mask(size).to(device)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Performs the forward pass of the model.
        """
        B, h, w = x.shape
        device = x.device
        seq_len = h * w

        # Flatten the image from (B, h, w) to (B, seq_len)
        x_flat = x.view(B, seq_len)

        # --- Auto-regressive shifting ---
        # 1. Embed the integer tokens into vectors
        # Use token 0 for the "start" position, as it's typically a special padding/start ID
        # or the first actual token of the sequence.
        # We rely on the shifting to model the auto-regressive property.
        embedded_tokens = self.token_embedding(x_flat)
        
        # 2. Shift the sequence to the right by prepending a zero vector for the first position
        # and removing the last token's embedding.
        # This makes the prediction for position `i` depend only on tokens up to `i-1`.
        # The model's first prediction (for the first pixel) will be based on the zero vector
        # and the positional embedding for position 0.
        input_sequence = torch.cat([torch.zeros(B, 1, self.d_latent, device=device), embedded_tokens[:, :-1, :]], dim=1)

        # Add positional embeddings
        # Ensure positional embedding matches the sequence length
        input_with_pos = input_sequence + self.pos_embedding[:, :seq_len, :]

        # Create the causal mask to prevent attending to future tokens
        causal_mask = self._generate_causal_mask(seq_len, device)

        # Pass through the transformer encoder
        transformer_output = self.transformer_encoder(input_with_pos, mask=causal_mask)

        # Project the output to get logits for each token in the vocabulary
        logits = self.output_head(transformer_output)

        # Reshape the output logits back to the image dimensions (B, h, w, n_tokens)
        logits_reshaped = logits.view(B, h, w, self.n_tokens)

        # The cross-entropy loss will be calculated by the trainer on these logits
        return logits_reshaped, {}

    @torch.no_grad()
    def generate(self, B: int = 1, h: int = 20, w: int = 30, device=None) -> torch.Tensor:
        """
        Generates new images token by token.
        """
        self.eval() # Set model to evaluation mode
        if device is None:
            device = next(self.parameters()).device
            
        seq_len = h * w
        # Start with an empty tensor for the output images, filled with zeros
        # We initialize with a value (e.g., 0) that will be embedded and shifted
        generated_tokens = torch.zeros((B, seq_len), dtype=torch.long, device=device)

        temperature = 1.5 # Ensure temperature is used for sampling
        
        # Iteratively generate one token at a time
        for i in range(seq_len):
            # Pass the currently generated sequence through the model
            # We reshape to (B, h, w) as expected by the forward pass
            current_sequence_img = generated_tokens.view(B, h, w)
            
            # The forward pass will automatically handle the shifting:
            # - For i=0, it will use the zero-vector placeholder + positional embedding for pos 0.
            # - For i>0, it will use the previously generated tokens up to i-1.
            logits, _ = self.forward(current_sequence_img)
            
            # Get the logits for the very next token we want to predict (position i)
            # Logits shape: (B, h, w, n_tokens) -> flatten to (B, seq_len, n_tokens)
            logits_flat = logits.view(B, seq_len, self.n_tokens)
            next_token_logits = logits_flat[:, i, :] # Get logits for the i-th position
            
            # Apply temperature and sample stochastically
            if temperature == 0:
                # Greedy decoding (deterministic)
                next_token = torch.argmax(next_token_logits, dim=-1)
            else:
                # Sample probabilistically (stochastic)
                probabilities = F.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1).squeeze(1)
            
            # Place the predicted token into our sequence
            generated_tokens[:, i] = next_token
        
        # Reshape the final sequence of tokens into image format
        return generated_tokens.view(B, h, w)