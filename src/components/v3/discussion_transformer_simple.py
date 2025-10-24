"""Modules related to the primary DiscussionTransformer model.

This model relates to the MDT model as presented in

Hebert, L., Sahu, G., Guo, Y., Sreenivas, N. K., Golab, L., & Cohen, R. (2024).
Multi-Modal Discussion Transformer: Integrating Text, Images and Graph
Transformers to Detect Hate Speech on Social Media.
Proceedings of the AAAI Conference on Artificial Intelligence,
38(20), 22096-22104. https://doi.org/10.1609/aaai.v38i20.30213
"""

from typing import Callable

import torch
import torch.nn as nn
import sentence_transformers
from sentence_transformers.models import Pooling
import transformers
import os

from torch.nn.attention.flex_attention import _DEFAULT_SPARSE_BLOCK_SIZE

from components.v2.graph_encoder_layer import BaseGraphTransformer
from components.v3 import input_utils
from utils.pylogger import RankedLogger

logger = RankedLogger(rank_zero_only=True)


class GraphStackLayer(torch.nn.Module):

    def __init__(
        self,
        pooling: Pooling,
        graph_encoder: BaseGraphTransformer,
        calculate_loss: bool = True,
        use_out_degree_emb: bool = False,
        num_out_degree: int = 10,
        out_degree_emb_dim: int = 32,
    ):
        super().__init__()
        self.graph_encoder = graph_encoder
        self.layer_norm = torch.nn.LayerNorm(768)
        self.pooling = pooling
        self.do_calculate_loss = calculate_loss
        self.use_out_degree_emb = use_out_degree_emb
        if self.use_out_degree_emb:
            self.out_degree_emb = nn.Embedding(
                num_embeddings=num_out_degree,
                embedding_dim=out_degree_emb_dim,
                padding_idx=0,
            )
            torch.nn.init.xavier_uniform_(self.out_degree_emb.weight)

    def get_node_token(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ):
        res = self.pooling.forward(
            {
                "token_embeddings": hidden_states,
                "attention_mask": attention_mask,
            }
        )
        return res["sentence_embedding"].type_as(hidden_states)

    def insert_graph_context(
        self,
        text_hidden_states: torch.Tensor,
        graph_node_embeddings: torch.Tensor,
        graph_token_position: torch.Tensor,
    ):
        # batch_size, _, _ = text_hidden_states.shape
        # rows = torch.arange(batch_size, device=text_hidden_states.device)
        text_hidden_states[:, graph_token_position, :] = graph_node_embeddings
        return text_hidden_states

    def calculate_recreation_loss(
        self,
        source_embeddings: torch.Tensor,
        created_embeddings: torch.Tensor,
        total_nodes: int | None = None,
    ):
        src = source_embeddings.detach()
        pred = created_embeddings
        if total_nodes is not None:
            src = src[:total_nodes]
            pred = pred[:total_nodes]

        # Directional alignment
        # src_n = torch.nn.functional.normalize(src, dim=-1)
        # pred_n = torch.nn.functional.normalize(pred, dim=-1)
        cos_loss = (
            1.0 - torch.nn.functional.cosine_similarity(src, pred, dim=-1)
        ).mean()

        # # Magnitude matching (scale-sensitive)
        # norm_loss = torch.nn.functional.mse_loss(
        #     pred.norm(dim=-1), src.norm(dim=-1)
        # )
        return cos_loss

    def forward(
        self,
        text_model_inputs: dict[str, torch.Tensor | bool],
        graph_mask: torch.Tensor,
        rope_spatial_pos: torch.Tensor,
        graph_token_position: torch.Tensor,
        original_attention_mask: torch.Tensor,
        out_degree: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        total_nodes: int | None = None,
    ):
        assert isinstance(text_model_inputs.get("hidden_states"), torch.Tensor)
        hidden_states: torch.Tensor = text_model_inputs["hidden_states"]
        original_node_embeddings = self.get_node_token(
            hidden_states, original_attention_mask
        )
        if self.use_out_degree_emb:
            assert out_degree is not None
            out_degree_embeddings = self.out_degree_emb(
                out_degree.clamp(max=self.out_degree_emb.num_embeddings - 1)
            )
            original_node_embeddings = (
                original_node_embeddings + out_degree_embeddings
            )

        graph_node_embeddings = self.graph_encoder(
            x=original_node_embeddings,
            mask=graph_mask,
            rope_spatial_pos=rope_spatial_pos,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        if self.do_calculate_loss:
            recreation_loss = self.calculate_recreation_loss(
                original_node_embeddings,
                graph_node_embeddings,
                total_nodes=total_nodes,
            )
        else:
            recreation_loss = torch.tensor(0.0, device=hidden_states.device)

        hidden_states = self.insert_graph_context(
            hidden_states, graph_node_embeddings, graph_token_position
        )
        text_model_inputs["hidden_states"] = hidden_states

        return text_model_inputs, recreation_loss


class TextStackLayer(torch.nn.Module):
    def __init__(self, gemma_layers: torch.nn.ModuleList):
        super().__init__()
        self.gemma_layers = gemma_layers

    def forward(
        self,
        text_model_inputs: dict[
            str, torch.Tensor | bool | dict[str, torch.Tensor]
        ],
    ):
        assert isinstance(text_model_inputs.get("hidden_states"), torch.Tensor)

        causal_mask_mapping: dict[str, torch.Tensor] = text_model_inputs[
            "causal_mask_mapping"
        ]
        for layer in self.gemma_layers:
            layer: transformers.gemma.modeling_gemma.GemmaDecoderLayer
            gemma_out = layer(
                **text_model_inputs,
                attention_mask=causal_mask_mapping[layer.attention_type],
            )
            text_model_inputs["hidden_states"] = gemma_out[0]

        return text_model_inputs


class CombinedBlock(torch.nn.Module):
    def __init__(
        self, text_layer: TextStackLayer, graph_layer: GraphStackLayer
    ):
        super().__init__()
        self.text_layer = text_layer
        self.graph_layer = graph_layer
        self.do_calculate_loss = graph_layer.do_calculate_loss

    def forward(
        self,
        text_model_inputs: dict[
            str, torch.Tensor | bool | dict[str, torch.Tensor]
        ],
        graph_token_position: torch.Tensor,
        rope_spatial_pos: torch.Tensor,
        graph_mask: torch.Tensor,
        original_attention_mask: torch.Tensor,
        out_degree: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        total_nodes: int | None = None,
    ):
        text_model_inputs = self.text_layer(text_model_inputs)
        text_model_inputs, recreation_loss = self.graph_layer(
            text_model_inputs=text_model_inputs,
            graph_mask=graph_mask,
            rope_spatial_pos=rope_spatial_pos,
            graph_token_position=graph_token_position,
            original_attention_mask=original_attention_mask,
            out_degree=out_degree,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            total_nodes=total_nodes,
        )
        return text_model_inputs, recreation_loss


def freeze_module_params(m: nn.Module, freeze: bool = True):
    """Given a module, freeze all of its parameters.

    Args:
        m (nn.Module): Module to freeze
    """
    logger.info(f"Setting grad for {m.__class__.__name__} to {not freeze}")
    if m is not None:
        for p in m.parameters():
            p.requires_grad = not freeze


class DiscussionTransformerV3(nn.Module):

    encoder: transformers.Gemma3TextModel
    blocks: nn.ModuleList
    last_text_layer: TextStackLayer
    output_pooling: sentence_transformers.SentenceTransformer

    def __init__(
        self,
        graph_stack_factory: Callable[[int], BaseGraphTransformer],
        num_out_degree: int,
        embedding_dim: int = 768,
        block_size: int = _DEFAULT_SPARSE_BLOCK_SIZE,
        use_out_degree_emb: bool = False,
    ) -> None:
        """The Discussion Transformer model, which fuses comment modalities with
        graph context.

        The model can largely be broken up into three parts, the initial
        embeddings, the bottleneck layers and the Graphormer layers.

        First - Initial Embeddings: We encode the content of the comments using
        the Text and Vision models described by the `text_model_config` and
        `vit_model_config`. We use the first `total_layers - fusion_layers`
        layers of each modality model.

        Second - Bottleneck Layers: We introduce bottleneck tokens
        (`num_bottle_neck`) to both the text and image inputs, making the inputs

            [B_1] [B_2] ... [B_n] [SEP] [T_1] [T_2] ... [T_n] [CLS]

        where [B_i] are the bottleneck tokens, [T_i] are the modality tokens.
        Then, each fusion layer computes a embedding for each token in the
        input, including the bottle neck tokens (producing B^T_i and B^I_i, for
        text and images).

        We then take the average of B^T and B^I tokens to produce
        the set of B tokens for the next layer. This process allows information
        to be shared between the two modality models, where each model is forced
        to compress relevant information into those tokens. NOTE: When there is
        no image input, the B tokens consist of only B^T tokens.

        Third - Graphormer Layers: After num_fusion_stack fusion layers, we then
        include graph context using graphormer layers. This is done by using
        the B_1 token as the node embedding for each comment in the graph. This
        is in addition to a "Global" token that is used to represent the entire
        graph.

        After the graph layers, the new computed node embeddings, now with graph
        context, replaces the B_1 token in the input for the subsequent fusion
        layers. This reprocess repeats until all fusion stacks are processed.

        The final layer of the model is a Graphormer layer. We then return
        both B_1 for each node and the global token as the output of the model.

        Args:
            graph_node_feature (GraphNodeFeature): Module to compute initial
                node features, such as in-degree and out-degree embeddings, and
                to add additional auxiliary graph tokens, such as learned global
                tokens.
            graph_stack_config (DictConfig): Configuration dict for building the
                Graphormer layer stack. See build_graphormer_graph_encoder_layer
                for more details.
            text_model_config (DictConfig): Configuration dict for building the
                Text Transformer model. See build_bert_encoder for more details.
            num_bottle_neck (int): Number of learned bottleneck tokens to use
                and append to the input of both models.
            num_fusion_stack (int): The number of fusion stack layers.
                fusion_stack_size * num_fusion_stack must be less then the
                number of ViT and Bert layers. Defaults to 1.
            fusion_stack_size (int, optional): Number of consecutive fusion
                layers in a stack. fusion_stack_size * num_fusion_stack must be
                less then the number of ViT and Bert layers. Defaults to 1.
            embedding_dim (int, optional): Global embedding dimension used by
                Graphromer and the modality models. Used to initialize the
                LayerNorm layer and for asserts. Defaults to 768.
            encoder_normalize_before (bool, optional): Whether to normalize the
                embeddings before the first graphormer layers. Defaults to
                False.
            embed_scale (float): Scalar to rescale the pre-graphormer
                embeddings. Defaults to 1.
            num_graph_layers_to_freeze (int, optional): Number of graphormer
                layers to freeze. Useful when fine-tuning a pre-trained
                checkpoint towards a different task. Defaults to 0.
            freeze_initial_encoders (bool, optional): Whether to freeze the
                pre-fusion layers of BERT and ViT. Defaults to False.
            graph_token_average (bool, optional): Whether to average the
                embeddings of the node tokens as the output. If False, we use the
                gCLS token instead. Defaults to False.
            block_size (int, optional): The sparse block size to use for attention
                masking. Defaults to _DEFAULT_SPARSE_BLOCK_SIZE (128).
        """
        super().__init__()

        local_rank = os.environ.get("SLURM_LOCALID")
        local_device = f"cuda:{local_rank}" if local_rank is not None else "cpu"

        checkpoint = sentence_transformers.SentenceTransformer(
            "google/embeddinggemma-300m",
            model_kwargs={
                "attn_implementation": "sdpa",
            },
            device="cpu",
        ).train()
        encoder: transformers.Gemma3TextModel = (
            transformers.AutoModel.from_pretrained(
                "google/embeddinggemma-300m",
                attn_implementation="sdpa",
                device_map="cpu",
            ).train()  # type: ignore
        )
        logger.warning("ENABLING GRAD FOR ENCODER")
        for param in encoder.parameters():
            param.requires_grad = True
        self.encoder_config = encoder.config
        self.encoder_embed_tokens = encoder.embed_tokens
        self.encoder_rotary_emb = encoder.rotary_emb
        self.encoder_rotary_emb_local = encoder.rotary_emb_local

        chunk_sizes = [5, 6, 6, 7]
        assert sum(chunk_sizes) == len(encoder.layers), (
            sum(chunk_sizes),
            len(encoder.layers),
        )

        layer_chunks = [
            encoder.layers[i : i + size]
            for i, size in zip(
                torch.cumsum(torch.tensor([0] + chunk_sizes[:-1]), 0),
                chunk_sizes,
            )
        ]

        pooling: Pooling = checkpoint[1]  # type: ignore

        self.blocks = torch.nn.ModuleList(
            [
                CombinedBlock(
                    TextStackLayer(layer_chunk),
                    GraphStackLayer(
                        pooling,
                        graph_encoder=graph_stack_factory(depth=i),
                        calculate_loss=True,
                        num_out_degree=num_out_degree,
                        out_degree_emb_dim=embedding_dim,
                        use_out_degree_emb=(
                            True if i == 0 and use_out_degree_emb else False
                        ),
                    ),
                )
                for i, layer_chunk in enumerate(layer_chunks[:-1])
            ]
        )
        self.last_text_layer = TextStackLayer(layer_chunks[-1])

        del checkpoint[0]  # Remove the encoder step
        for param in checkpoint.parameters():
            param.requires_grad = False
        self.output_pooling = checkpoint
        self.block_size = block_size

    def forward(
        self,
        text_input: dict[str, torch.Tensor],
        rotary_pos: torch.Tensor,
        spatial_pos: torch.Tensor,
        out_degree: torch.Tensor,
        num_total_graphs: int,
        graph_mask: torch.Tensor,
        graph_ids: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        num_total_nodes: int | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor] | None]:
        """The forward function of the Discussion Transformer model.

        Args:
            == Text inputs ==
            text_input (dict[str, torch.Tensor]): The tokenized text inputs,
                containing:
                - text_input_ids (torch.Tensor): batched tokenized text ids, with
                    shape (batch_size * nodes, T)
                - text_token_type_ids (torch.Tensor): batched token type ids, with
                    shape (batch_size * nodes, T)
                - text_attention_mask (torch.Tensor): batched text attention mask,
                    with shape (batch_size * nodes, T), where 1 indicates a token
                    that should be attended to and 0 indicates padding.

            == Graph inputs ==
            graph_ids (torch.Tensor): Id of the graph each node belongs to,
                where padding nodes are assigned the value PADDING_GRAPH_ID, with
                shape (batch_size * nodes). This is used to mask out attention.
                It is assumed that the graph_ids are contiguous and start from 0.
            spatial_pos (torch.Tensor): Matrix with shape
                (batch_size * nodes, batch_size * nodes, 2) indicating the
                number of up hops and down hops between each node in the graph.
            in_degree (torch.Tensor): batched in-degrees, corresponding to the
                in-degree of each node in the graph. Padded with 0s and shifted
                by 1. Shape (batch_size * nodes).
            out_degree (torch.Tensor): batched out-degrees, corresponding to the
                out-degree of each node in the graph. Padded with 0s and shifted
                by 1. Shape (batch_size * nodes).
            num_total_graphs (int): Total number of unique graphs in the batch,
                shape (). Note that this is different then graph_ids, which is
                node-wise.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Returns
                - node_embedding: The final node embeddings for each node in the
                    graph with shape (batch_size * nodes, C).
                - global_embedding: The final global embedding for the graph,
                    with shape (batch_size, C).
        """
        # Since we do not use global tokens, we remove them from graph_ids
        # graph_ids = graph_ids[num_total_graphs:]
        # graph_mask = graph_mask[:, :, num_total_graphs:, num_total_graphs:]
        # rotary_pos = rotary_pos[num_total_graphs:]

        prepared_text_inputs, graph_position, original_mask = (
            input_utils.prepare_input(
                self.encoder_config,
                self.encoder_embed_tokens,
                self.encoder_rotary_emb,
                self.encoder_rotary_emb_local,
                text_input["input_ids"],
                text_input["attention_mask"],
            )
        )
        cos_loss = torch.zeros(len(self.blocks), device=graph_ids.device)
        # norm_loss = torch.zeros(len(self.blocks), device=graph_ids.device)
        for i, block in enumerate(self.blocks):
            prepared_text_inputs, recreation_loss = block.forward(
                text_model_inputs=prepared_text_inputs,
                graph_token_position=graph_position,
                rope_spatial_pos=rotary_pos,
                graph_mask=graph_mask,
                original_attention_mask=original_mask,
                out_degree=out_degree,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                total_nodes=num_total_nodes,
            )
            # instance_cos, instance_norm = recreation_loss
            cos_loss[i] = recreation_loss
            # norm_loss[i] = instance_norm

        text_input = self.last_text_layer(prepared_text_inputs)
        output = self.output_pooling(
            {
                "token_embeddings": text_input["hidden_states"],
                "attention_mask": original_mask,
            }
        )
        output_sentence_embeddings = output["sentence_embedding"]
        global_embedding = self.average_embeddings_by_index(
            output_sentence_embeddings, graph_ids
        )
        node_embedding = output_sentence_embeddings
        return (
            node_embedding,
            global_embedding,
            {
                "cos_loss": cos_loss.mean(),
                "norm_loss": torch.tensor(0.0, device=cos_loss.device),
            },
        )

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
