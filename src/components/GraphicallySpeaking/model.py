import torch
from torch import nn
from torch_geometric.nn import GATConv, Sequential
from omegaconf import DictConfig
from transformers import AutoModel


class GraphicallySpeakingModel(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        text_model_config: DictConfig,
        block_size: int = 1,  # unused, but required for compatibility
        num_heads=8,
        dropout=0.4,
    ):
        super(GraphicallySpeakingModel, self).__init__()
        self.text_model = AutoModel.from_pretrained(
            text_model_config["bert_model_name"]
        )
        self.embedding_dim = embedding_dim
        self.model = Sequential(
            "x, edge_index",
            [
                (
                    GATConv(
                        embedding_dim,
                        hidden_dim,
                        heads=num_heads,
                        dropout=dropout,
                    ),
                    "x, edge_index -> x",
                ),
                (nn.ELU(), "x -> x"),
                (
                    GATConv(
                        hidden_dim * num_heads,
                        hidden_dim,
                        heads=num_heads,
                    ),
                    "x, edge_index -> x",
                ),
                (nn.ELU(), "x -> x"),
                (
                    GATConv(hidden_dim * num_heads, embedding_dim, heads=1),
                    "x, edge_index -> x",
                ),
            ],
        )
        self.fc = nn.Linear(embedding_dim * 2, embedding_dim)

    def forward(
        self, text_input: dict[str, torch.Tensor], edge_index: torch.Tensor
    ):
        text_cls = self.text_model(**text_input).pooler_output
        assert text_cls.shape[-1] == self.embedding_dim, (
            f"Text model output shape {text_cls.shape} does not match "
            f"embedding dimension {self.embedding_dim}"
        )
        x = self.model(text_cls, edge_index)
        assert x.shape[-1] == self.embedding_dim, (
            f"Graph model output shape {x.shape} does not match "
            f"embedding dimension {self.embedding_dim}"
        )
        x = torch.cat((x, text_cls), dim=1)
        x = self.fc(x)
        return x
