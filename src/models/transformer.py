import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.transformer_block import Block
from torchtitan.protocols.train_spec import BaseModelArgs
from torchtitan.tools.logging import logger

import math



class TransformerArgs(BaseModelArgs):
    """
    Arguments for the Transformer model.

    Args:
        n_head (int): Number of attention heads.
        n_embed (int): Embedding dimension.
        context_length (int): Maximum sequence length.
        vocab_size (int): Size of the vocabulary.
        N_BLOCKS (int): Number of transformer blocks.
    """
    def __init__(self, n_head: int, n_embed: int, context_length: int, vocab_size: int, N_BLOCKS: int) -> None:
        super().__init__()
        self.n_head = n_head
        self.n_embed = n_embed
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.N_BLOCKS = N_BLOCKS

    def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
        # TODO(jianiw): Add the number of flops for the autoencoder
        nparams = sum(p.numel() for p in model.parameters())
        nparams_embedding = sum(
            sum(p.numel() for p in m.parameters())
            for m in model.children()
            if isinstance(m, nn.Embedding)
        )

        l, h, q, t = (
            self.N_BLOCKS,
            self.n_head,
            self.n_embed // self.n_head,
            seq_len,
        )
        # Reasoning behind the factor of 12 for the self-attention part of the formula:
        # 1. each self-attention has 2 matmul in the forward and 4 in the backward (6)
        # 2. the flash attention does 1 more matmul recomputation in the backward
        #    but recomputation should not be counted in calculating MFU           (+0)
        # 3. each matmul performs 1 multiplication and 1 addition                 (*2)
        # 4. we follow the convention and do not account for sparsity in causal attention
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
        return nparams, num_flops_per_token
        



class Transformer(nn.Module):
    """
    The main Transformer model.

    This class combines token and position embeddings with a sequence of Transformer blocks
    and a final linear layer for language modeling.

    Args:
        n_head (int): The number of attention heads in each transformer block.
        n_embed (int): The dimensionality of the embedding space.
        context_length (int): The maximum length of the input sequence.
        vocab_size (int): The size of the vocabulary.
        N_BLOCKS (int): The number of transformer blocks in the model.
    """
    def __init__(self, model_args: TransformerArgs) -> None:
        """
        Initializes the Transformer model.

        Args:
            n_head (int): Number of attention heads.
            n_embed (int): Embedding dimension.
            context_length (int): Maximum sequence length.
            vocab_size (int): Size of the vocabulary.
            N_BLOCKS (int): Number of transformer blocks.
        """
        super().__init__()
        self.context_length = model_args.context_length
        self.N_BLOCKS = model_args.N_BLOCKS
        n_head = model_args.n_head
        n_embed = model_args.n_embed
        vocab_size = model_args.vocab_size
        
        self.token_embed = nn.Embedding(vocab_size, n_embed)
        self.position_embed = nn.Embedding(self.context_length, n_embed)
        self.attn_blocks = nn.ModuleList([Block(n_head, n_embed, self.context_length) for _ in range(self.N_BLOCKS)])
        self.layer_norm = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.register_buffer('pos_idxs', torch.arange(self.context_length))

        
    def init_weights(self, buffer_device=None):
        """
        Initializes the weights of the Transformer model.
        This method applies a standard initialization strategy:
        - nn.Linear layers: Normal distribution (mean=0, std=0.02) for weights, zeros for biases.
        - nn.Embedding layers: Normal distribution (mean=0, std=0.02) for weights.
        - nn.LayerNorm layers: Ones for weights (gamma), zeros for biases (beta).
        It also applies a special scaling for the weights of the final linear layer in each MLP block.
        """
        self.apply(self._init_module_weights)

        # Apply special scaling to the weights of the second linear layer (fc2) 
        # in MLP blocks. This is a common practice in GPT-like models,
        # where N_BLOCKS is the number of residual layers.
        for module_name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # Check if this linear layer is the second one in an MLP (fc2)
                # The name would be like 'attn_blocks.<index>.mlp.fc2'
                if 'mlp.fc2' in module_name:
                    torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.N_BLOCKS))
                    # Biases are already initialized to zero by _init_module_weights
                    # if module.bias is not None:
                    #     torch.nn.init.zeros_(module.bias)
    
    @staticmethod
    def _init_module_weights(module):
        """
        Helper function to initialize weights for different module types.
        Applied recursively by model.apply().
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            # LayerNorm has learnable affine parameters (weight, bias) by default (elementwise_affine=True)
            if hasattr(module, 'weight') and module.weight is not None: # Check if elementwise_affine is True
                torch.nn.init.ones_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None: # Check if elementwise_affine is True
                torch.nn.init.zeros_(module.bias)

    def _pre_attn_pass(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Combines token and position embeddings.

        Args:
            idx (torch.Tensor): Input token indices.

        Returns:
            torch.Tensor: Sum of token and position embeddings.
        """
        B, T = idx.shape
        tok_embedding = self.token_embed(idx)
        pos_embedding = self.position_embed(self.pos_idxs[:T])
        return tok_embedding + pos_embedding

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass through the Transformer.

        Args:
            idx (torch.Tensor): Input token indices.
            targets (torch.Tensor, optional): Target token indices for loss calculation. Defaults to None.

        Returns:
            tuple: Logits and loss (if targets are provided).
        """
        x1 = self._pre_attn_pass(idx)
        for block in self.attn_blocks:
            x1 = block(x1)
        x2 = self.layer_norm(x1)
        logits = self.lm_head(x2)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            flat_logits = logits.view(B * T, C)
            targets = targets.reshape(B * T).long()
            loss = F.cross_entropy(flat_logits, targets)
        return logits, loss

    def forward_embedding(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass focusing on the embedding and attention blocks.

        Args:
            idx (torch.Tensor): Input token indices.

        Returns:
            tuple: Output after attention blocks and the residual.
        """
        x = self._pre_attn_pass(idx)
        residual = x
        for block in self.attn_blocks:
            x, residual = block.forward_embedding(x)
        return x, residual

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        Generates new tokens given a starting sequence.

        Args:
            idx (torch.Tensor): Initial sequence of token indices.
            max_new_tokens (int): Number of tokens to generate.

        Returns:
            torch.Tensor: The extended sequence of tokens.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context_length:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    @classmethod
    def from_model_args(cls, model_args: TransformerArgs) -> "Transformer":
        """
        Initialize a Flux model from a FluxModelArgs object.

        Args:
            model_args (FluxModelArgs): Model configuration arguments.

        Returns:
            FluxModel: FluxModel model.

        """
        return cls(model_args)

if __name__ == '__main__':
    # Example Usage (optional, for testing the module independently)
    batch_size = 2
    sequence_length = 5
    vocab_size = 100
    embedding_dim = 32
    num_heads = 4
    num_blocks = 2
    context_len = 5
    input_indices = torch.randint(0, vocab_size, (batch_size, sequence_length))

    transformer_model = Transformer(n_head=num_heads, n_embed=embedding_dim, context_length=context_len, vocab_size=vocab_size, N_BLOCKS=num_blocks)
    logits, loss = transformer_model(input_indices, targets=input_indices) # Using input as target for simplicity

    print("Transformer Logits Shape:", logits.shape)
    print("Transformer Loss:", loss)

    # Example of generating tokens
    start_indices = input_indices[:, :1]  # Take the first token of each sequence as start
    generated_tokens = transformer_model.generate(start_indices, max_new_tokens=5)
    print("Generated Tokens Shape:", generated_tokens.shape)