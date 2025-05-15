import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoderLayer(nn.Module):
    """
    Implements a standard Transformer Encoder Layer as described in
    "Attention Is All You Need".

    Consists of two sub-layers:
    1. Multi-Head Self-Attention followed by Add & Norm.
    2. Position-wise Feed-Forward Network followed by Add & Norm.
    """
    def __init__(self, embedding_dim, num_heads, dropout):
        super(TransformerEncoderLayer, self).__init__()

        # --- Sub-layer 1: Multi-Head Self-Attention ---
        self.attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        # Dropout layer after attention output, before residual connection
        self.dropout1 = nn.Dropout(dropout)
        # Layer Normalization after the first sub-layer (Attention + Add)
        self.norm1 = nn.LayerNorm(embedding_dim)

        # --- Sub-layer 2: Position-wise Feed-Forward Network ---
        self.feedforward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4), # Expansion layer
            nn.GELU(),                                   # Activation (GELU is common, ReLU also used)
            nn.Dropout(dropout),                         # Dropout within FFN
            nn.Linear(embedding_dim * 4, embedding_dim)  # Projection back to embedding_dim
        )
        # Dropout layer after feedforward output, before residual connection
        self.dropout2 = nn.Dropout(dropout)
        # Layer Normalization after the second sub-layer (FFN + Add)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, query, key=None, value=None, src_mask=None, src_key_padding_mask=None):
        """
        Forward pass for the Transformer Encoder Layer.

        Args:
            query (torch.Tensor): Input tensor (or query for cross-attention).
                                  Shape: (batch_size, sequence_length, embedding_dim)
            key (torch.Tensor, optional): Key tensor for attention. Defaults to query.
                                          Shape: (batch_size, key_sequence_length, embedding_dim)
            value (torch.Tensor, optional): Value tensor for attention. Defaults to query.
                                            Shape: (batch_size, value_sequence_length, embedding_dim)
            src_mask (torch.Tensor, optional): Mask to prevent attention to certain positions
                                               (e.g., future tokens in decoder). Shape: (sequence_length, sequence_length) or others broadcastable.
            src_key_padding_mask (torch.Tensor, optional): Mask to indicate padding tokens in the key sequence.
                                                           Shape: (batch_size, key_sequence_length)

        Returns:
            torch.Tensor: Output tensor of the layer.
                          Shape: (batch_size, sequence_length, embedding_dim)
        """
        # Default to self-attention if key and value not provided
        if key is None and value is None:
            key = value = query

        # 1. Multi-Head Attention sub-layer
        # Pass query, key, value, and optional masks to MultiheadAttention
        attn_output ,_ = self.attn(
            query, key, value,
            attn_mask=src_mask,            # Mask for preventing attention to certain positions
            key_padding_mask=src_key_padding_mask, # Mask for ignoring padding tokens
            need_weights=False             # Usually we don't need weights outside the layer
        )

        # 2. Add & Norm (after Attention)
        # Apply dropout to the attention output, add the residual connection (input query),
        # and then apply Layer Normalization.
        # x becomes the input for the next sub-layer
        x = self.norm1(query + self.dropout1(attn_output))

        # 3. Feed Forward sub-layer
        ff_output = self.feedforward(x)

        # 4. Add & Norm (after Feed Forward)
        # Apply dropout to the feedforward output, add the residual connection (output from previous Add & Norm),
        # and then apply Layer Normalization.
        out = self.norm2(x + self.dropout2(ff_output))

        return out