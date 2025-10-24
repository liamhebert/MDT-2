import torch
import transformers
from torch import nn
import sentence_transformers


class TextOnlyModel(nn.Module):
    def __init__(self, block_size: int = 256):
        super(TextOnlyModel, self).__init__()
        self.block_size = block_size
        checkpoint = sentence_transformers.SentenceTransformer(
            "google/embeddinggemma-300m",
            model_kwargs={
                "attn_implementation": "sdpa",
            },
            device="cpu",
        ).train()
        self.encoder: transformers.Gemma3TextModel = checkpoint[0]
        del checkpoint[0]  # Remove the transformer layer from the stack
        self.pooler = checkpoint

    def forward(
        self,
        text_input: dict[str, torch.Tensor],
        graph_ids: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor] | None]:
        mask = text_input.get("attention_mask", None)

        encoder_output = self.encoder(features=text_input)
        pooler_output = self.pooler(encoder_output)  # type: ignore
        embeddings = pooler_output["sentence_embedding"]
        global_embedding = self.average_embeddings_by_index(
            embeddings, graph_ids
        )
        return embeddings, global_embedding, {}

    def average_embeddings_by_index(
        self, embeddings: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Averages embeddings based on shared index values.

        Args:
            embeddings: A tensor of shape (B, E) representing the embeddings.
                B is the batch size (or number of embeddings), and E is the
                embedding dimension.
            indices: A tensor of shape (B) containing integer indices.  There
                are G unique values in this tensor.  Each index in `indices`
                corresponds to an embedding in `embeddings`.

        Returns:
            A tensor of shape (G, E) where each row represents the average
            embedding for all embeddings sharing the same index value. The order
            of the returned embeddings corresponds to the sorted order of the
            unique indices. Returns an empty tensor if the input is empty.
        """
        if embeddings.numel() == 0:
            return torch.empty(
                0,
                embeddings.size(1),
                dtype=embeddings.dtype,
                device=embeddings.device,
            )

        unique_indices, inverse_indices = torch.unique_consecutive(
            indices, return_inverse=True
        )
        num_unique = unique_indices.size(0)
        embedding_dim = embeddings.size(1)

        # Initialize the output tensor
        averaged_embeddings = torch.zeros(
            (num_unique, embedding_dim),
            dtype=embeddings.dtype,
            device=embeddings.device,
        )

        # Use scatter_reduce with "mean" to directly compute the average.
        averaged_embeddings = averaged_embeddings.scatter_reduce(
            0,
            inverse_indices.unsqueeze(1).expand(-1, embedding_dim),
            embeddings,
            reduce="mean",
            include_self=False,
        )

        return averaged_embeddings
