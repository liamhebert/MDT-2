"""Modules related to the primary DiscussionTransformer model.

This model relates to the MDT model as presented in

Hebert, L., Sahu, G., Guo, Y., Sreenivas, N. K., Golab, L., & Cohen, R. (2024).
Multi-Modal Discussion Transformer: Integrating Text, Images and Graph
Transformers to Detect Hate Speech on Social Media.
Proceedings of the AAAI Conference on Artificial Intelligence,
38(20), 22096-22104. https://doi.org/10.1609/aaai.v38i20.30213
"""

from typing import Tuple, Callable

from omegaconf import DictConfig
import torch
import torch.nn as nn
from transformers import AutoModel
from transformers import PretrainedConfig
from transformers.models.bert.modeling_bert import BertConfig
from transformers.models.bert.modeling_bert import BertLayer
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.modeling_bert import BertPooler
from transformers.models.vit.modeling_vit import ViTConfig
from transformers.models.vit.modeling_vit import ViTLayer
from transformers.models.vit.modeling_vit import ViTModel
from transformers.models.vit.modeling_vit import ViTPooler
from torch.nn.attention.flex_attention import _DEFAULT_SPARSE_BLOCK_SIZE
from abc import ABC, abstractmethod

from components.v2.feature_layers import GraphAttnBias
from components.v2.feature_layers import GraphNodeFeature
from components.v2.discussion_blocks import DiscussionTransformerBlock
from components.graph_fusion_layer import GraphFusionStack
from components.v2.graph_encoder_layer import BaseGraphTransformer

from utils.pylogger import RankedLogger

from data.types import TextFeatures

logger = RankedLogger(rank_zero_only=True)


def init_graphormer_params(module):
    """
    Initialize the weights specific to the Graphormer Model.
    """

    def normal_(data: nn.Parameter):
        """Sets tensor parameters to fit normal distribution.

        Args:
            data (nn.Parameter): Parameter to initialize to normal
        """
        # with FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()


def freeze_module_params(m: nn.Module, freeze: bool = True):
    """Given a module, freeze all of its parameters.

    Args:
        m (nn.Module): Module to freeze
    """
    logger.info(f"Setting grad for {m.__class__.__name__} to {not freeze}")
    if m is not None:
        for p in m.parameters():
            p.requires_grad = not freeze


class DiscussionTransformerPrototype(ABC, nn.Module):

    embedding_dim: int
    graph_node_feature: GraphNodeFeature
    vit_model: ViTModel
    text_model: BertModel
    first_fusion_layer: GraphFusionStack
    blocks: nn.ModuleList
    final_graphormer_layer: BaseGraphTransformer
    num_bottle_neck: int
    bottle_neck: nn.Parameter
    block_size: int

    def __init__(
        self,
        graph_node_feature: GraphNodeFeature,
        graph_stack_factory: Callable[[int], BaseGraphTransformer],
        vit_model_config: DictConfig,
        text_model_config: DictConfig,
        num_bottle_neck: int,
        num_fusion_stack: int = 1,
        fusion_stack_size: int = 1,
        embedding_dim: int = 768,
        num_graph_layers_to_freeze: int = 0,
        freeze_initial_encoders: bool = False,
        graph_token_average: bool = False,
        block_size: int = _DEFAULT_SPARSE_BLOCK_SIZE,
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
            graph_attn_bias (GraphAttnBias): Module to compute attention bias
                added during graph attention. This uses the distance between
                nodes to add a bias scalar to the attention weights (one for
                each head).
            graph_stack_config (DictConfig): Configuration dict for building the
                Graphormer layer stack. See build_graphormer_graph_encoder_layer
                for more details.
            vit_model_config (DictConfig): Configuration dict for building the
                Vision Transformer model.
                See build_vit_encoder for more details.
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

        # TODO(liamhebert): It might be worth having a pass-through for
        # no-fusion models, where we just use graph layers.
        assert num_fusion_stack > 0, "num_fusion_stack must be greater than 0"
        assert fusion_stack_size > 0, "fusion_stack_size must be greater than 0"

        self.embedding_dim = embedding_dim
        self.graph_node_feature = graph_node_feature
        self.graph_token_average = graph_token_average

        total_fusion_layers = fusion_stack_size * num_fusion_stack
        self.vit_model, vit_fusion_layers = self.build_vit_encoder(
            num_fusion_layers=total_fusion_layers,
            **vit_model_config,
        )  # type: ignore[misc]

        self.text_model, text_fusion_layers = self.build_bert_encoder(
            num_fusion_layers=total_fusion_layers,
            **text_model_config,
        )  # type: ignore[misc]

        if freeze_initial_encoders:
            freeze_module_params(self.text_model, freeze=True)
            freeze_module_params(self.vit_model, freeze=True)

        freeze_module_params(text_fusion_layers, freeze=False)
        freeze_module_params(vit_fusion_layers, freeze=False)

        assert len(text_fusion_layers) == len(vit_fusion_layers)
        assert (
            len(text_fusion_layers) == fusion_stack_size * num_fusion_stack
        ), f"Expected {total_fusion_layers=}, got {len(text_fusion_layers)=}"

        grouped_text_layers, grouped_vit_layers = self.group_layers_for_fusion(
            text_fusion_layers, vit_fusion_layers, fusion_stack_size
        )

        self.first_fusion_layer = GraphFusionStack(
            grouped_text_layers[0],
            grouped_vit_layers[0],
            bert_dim=self.embedding_dim,
            vit_dim=self.embedding_dim,
            bottleneck_dim=self.embedding_dim,
            use_projection=True,
        )

        self.blocks = nn.ModuleList(
            [
                self.build_discussion_block(
                    graph_layer=graph_stack_factory(depth=i),  # type: ignore
                    bert_layer_stack=text,
                    vit_layer_stack=vit,
                    embedding_dim=self.embedding_dim,
                    num_bottlenecks=num_bottle_neck,
                )
                for i, (text, vit) in enumerate(
                    zip(grouped_text_layers[1:], grouped_vit_layers[1:])
                )
            ]
        )

        self.final_graphormer_layer = graph_stack_factory(
            depth=len(self.blocks) + 1  # type: ignore
        )

        assert (
            len(self.blocks) + 1 == num_fusion_stack
        ), f"Expected {num_fusion_stack=}, got {len(self.blocks) + 1=}"

        self.num_bottle_neck = num_bottle_neck
        self.bottle_neck = nn.Parameter(
            torch.zeros(
                (num_bottle_neck, self.embedding_dim), dtype=torch.float
            ).normal_(mean=0.0, std=0.02),
            requires_grad=True,
        )
        self.block_size = block_size

        for layer in range(num_graph_layers_to_freeze):
            freeze_module_params(self.graphormer_layers[layer])

    def group_layers_for_fusion(
        self,
        text_fusion_layers: list[BertLayer],
        vit_fusion_layers: list[ViTLayer],
        fusion_stack_size: int,
    ) -> tuple[list[list[BertLayer]], list[list[ViTLayer]]]:
        """Fuses the lists of modality layers into a list of fusion layers.

        Here, we group the layers into groups of `fusion_stack_size` layers,
        where each layer will process the output of the previous layer.

        NOTE: the number of text and vision layers must be equal and divisible
        by `fusion_stack_size`.

        Args:
            text_fusion_layers (list[BertLayer]): List of text layers to fuse.
            vit_fusion_layers (list[ViTLayer]): List of vision layers to fuse.
            fusion_stack_size (int): Number of consecutive fusion layers to
                stack.

        Returns:
            list[list[BertLayer]], list[list[ViTLayer]]: Grouped lists of layers
                to be used for fusion.
        """

        # TODO(liamhebert): Consider using itertools.zip_longest to handle
        # uneven lengths
        grouped_text_fusion_layers = [
            text_fusion_layers[i : i + fusion_stack_size]
            for i in range(
                0,
                len(text_fusion_layers),
                fusion_stack_size,
            )
        ]
        grouped_vit_fusion_layers = [
            vit_fusion_layers[i : i + fusion_stack_size]
            for i in range(
                0,
                len(vit_fusion_layers),
                fusion_stack_size,
            )
        ]
        assert len(grouped_text_fusion_layers) == len(grouped_vit_fusion_layers)

        # fusion_layers = nn.ModuleList(
        #     [
        #         GraphFusionStack(text_layer, vit_layer, use_projection=False)
        #         for text_layer, vit_layer in zip(
        #             grouped_text_fusion_layers, grouped_vit_fusion_layers
        #         )
        #     ]
        # )

        return grouped_text_fusion_layers, grouped_vit_fusion_layers

    def build_vit_encoder(
        self,
        vit_model_name: str,
        attention_dropout: float,
        activation_dropout: float,
        num_fusion_layers: int,
        test_config: ViTConfig | None = None,
    ) -> Tuple[ViTModel, list[ViTLayer]]:
        """Constructs the Vision Transformer model, taking the last
        `num_fusion_layers` layers for fusion.

        Args:
            vit_model_name (str): The huggingface model name to use.
            attention_dropout (float): The dropout probability for attention
            activation_dropout (float): The dropout probability for the
                activations
            num_fusion_layers (int): The number of layers to leave out for
                fusion, taking from the end of the model stack.
            test_config (ViTConfig, optional): If set, manual test_config to use
                for testing. Should not be used in production. Defaults to None.

        Returns:
            Tuple[ViTModel, list[ViTLayer]]: A tuple conisting of
                - The Vision Transformer model with the last `num_fusion_layers`
                    layers removed.
                - The removed layers, to be used for fusion.
        """
        vit_model: ViTModel

        if test_config is not None:
            print(
                type(test_config), not isinstance(test_config, PretrainedConfig)
            )
            print(type(test_config), isinstance(test_config, PretrainedConfig))

            logger.warning(
                "Using test config for ViT model."
                "If this is production, please fix!"
            )
            vit_model = ViTModel(config=test_config)
        else:
            vit_model = AutoModel.from_pretrained(
                vit_model_name,
                attn_implementation="sdpa",
                # hidden_dropout_prob=activation_dropout,
                # attention_probs_dropout_prob=attention_dropout,
            ).train()
            if hasattr(vit_model, "vision_model"):
                vit_model = vit_model.vision_model

        if num_fusion_layers == 0:
            vit_other_layers = []
        else:
            num_fusion_layers = num_fusion_layers
            encoder = vit_model.encoder
            if hasattr(encoder, "layer"):
                vit_other_layers = encoder.layer[-num_fusion_layers:]
                encoder.layer = encoder.layer[:-num_fusion_layers]
            elif hasattr(encoder, "layers"):
                vit_other_layers = encoder.layers[-num_fusion_layers:]
                encoder.layers = vit_model.encoder.layers[:-num_fusion_layers]
            else:
                raise ValueError("Unknown encoder type for ViT model.")

        return vit_model, vit_other_layers

    def build_bert_encoder(
        self,
        bert_model_name: str,
        attention_dropout: float,
        activation_dropout: float,
        num_fusion_layers: int,
        test_config: BertConfig | None = None,
    ) -> Tuple[BertModel, list[BertLayer]]:
        """Constructs the Text Transformer model, taking the last
        `num_fusion_layers` layers for fusion.

        Args:
            bert_model_name (str): The huggingface model name to use
            attention_dropout (float): The dropout probability for attention
            activation_dropout (float): The dropout probability for the
                activations
            num_fusion_layers (int): The number of layers to leave out for
                fusion, taking from the end of the model stack.
            test_config (BertConfig, optional): If set, manual test_config to
                use for testing. Should not be used in production. Defaults to
                None.

        Returns:
            Tuple[BertModel, list[BertLayer], BertPooler]: A tuple conisting of
                - The Bert Transformer model with the last `num_fusion_layers`
                    layers removed.
                - The removed layers, to be used for fusion.
        """
        # TODO(liamhebert): Try to fuse the two build_(bert|vit) functions
        # together since they have the same logic, the only difference is the
        # bert_model call.
        bert: BertModel

        if test_config is not None:
            logger.warning(
                "Using test config for BERT model."
                "If this is production, please fix!"
            )
            bert = BertModel(test_config)
        else:
            bert = AutoModel.from_pretrained(
                bert_model_name,
                attn_implementation="sdpa",
                # hidden_dropout_prob=activation_dropout,
                # attention_probs_dropout_prob=attention_dropout,
            ).train()
            if hasattr(bert, "text_model"):
                bert = bert.text_model

        if num_fusion_layers == 0:
            bert_other_layers = []
        else:
            num_fusion_layers = num_fusion_layers
            encoder = bert.encoder
            if hasattr(encoder, "layer"):
                bert_other_layers = encoder.layer[-num_fusion_layers:]
                encoder.layer = encoder.layer[:-num_fusion_layers]
            elif hasattr(encoder, "layers"):
                bert_other_layers = encoder.layers[-num_fusion_layers:]
                encoder.layers = bert.encoder.layers[:-num_fusion_layers]
            else:
                raise ValueError("Unknown encoder type for BERT model.")

        return bert, bert_other_layers

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

        unique_indices, inverse_indices = torch.unique(
            indices, return_inverse=True, sorted=True
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
        assert False
        return averaged_embeddings

    @abstractmethod
    def forward(
        self,
        text_input: dict[str, torch.Tensor],
        image_input: dict[str, torch.Tensor],
        image_padding_mask: torch.Tensor,
        rotary_pos: torch.Tensor,
        out_degree: torch.Tensor,
        num_total_graphs: int,
        graph_mask: torch.Tensor,
        graph_ids: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

            == Image inputs ==
            image_input (torch.Tensor): batched and tokenized image features,
                with shape (batch_size * nodes, D)
            image_padding_mask (torch.Tensor): batched boolean tensor indicating
                whether a node has an image. Shape (batch_size * nodes).

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
        pass

    @abstractmethod
    def build_discussion_block(
        self,
        graph_layer: BaseGraphTransformer,
        bert_layer_stack: list[BertLayer],
        vit_layer_stack: list[ViTLayer],
        embedding_dim: int,
        num_bottlenecks: int,
    ) -> DiscussionTransformerBlock: ...


class DiscussionTransformer(DiscussionTransformerPrototype):
    """
    Primary model class to create discussion and comment embeddings.
    """

    embedding_dim: int
    graph_node_feature: GraphNodeFeature
    graph_attn_bias: GraphAttnBias
    vit_model: ViTModel
    vit_pooler: ViTPooler
    text_model: BertModel
    text_pooler: BertPooler
    fusion_layers: nn.ModuleList
    graphormer_layers: nn.ModuleList
    num_bottle_neck: int
    bottle_neck: nn.Parameter

    def build_discussion_block(
        self,
        graph_layer: BaseGraphTransformer,
        bert_layer_stack: list[BertLayer],
        vit_layer_stack: list[ViTLayer],
        embedding_dim: int,
        num_bottlenecks: int,
    ):
        """Returns a DiscussionTransformerBlock with the given parameters."""
        return DiscussionTransformerBlock(
            graph_layer=graph_layer,
            bert_layer_stack=bert_layer_stack,
            vit_layer_stack=vit_layer_stack,
            embedding_dim=embedding_dim,
            num_bottlenecks=num_bottlenecks,
        )

    def forward(
        self,
        text_input: dict[str, torch.Tensor],
        image_input: dict[str, torch.Tensor],
        image_padding_mask: torch.Tensor,
        rotary_pos: torch.Tensor,
        out_degree: torch.Tensor,
        num_total_graphs: int,
        graph_ids: torch.Tensor,
        graph_mask: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

            == Image inputs ==
            image_input (torch.Tensor): batched and tokenized image features,
                with shape (batch_size * nodes, D)
            image_padding_mask (torch.Tensor): batched boolean tensor indicating
                whether a node has an image. Shape (batch_size * nodes).

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
        # Ideally, we keep a consistent shape and just flatten here, rather then
        # having to dynamically index here. Attention mask should handle this.

        bert_output = self.text_model(**text_input).last_hidden_state
        text_attention_mask = text_input[TextFeatures.AttentionMask]

        flattened_batch, seq_len, hidden_dim = bert_output.size()

        if image_input["pixel_values"].shape[0] > 0:
            vit_output = self.vit_model(**image_input).last_hidden_state
        else:
            vit_output = None

        bottle_neck = self.bottle_neck.repeat(flattened_batch, 1, 1)

        bert_output, vit_output, bottle_neck = self.first_fusion_layer(
            bert_hidden_states=bert_output,
            vit_hidden_states=vit_output,
            bottle_neck=bottle_neck,
            image_padding_mask=image_padding_mask,
            bert_attention_mask=text_attention_mask,
        )
        # This does not have the graph ids in them
        graph_x = bottle_neck[:, 0, :]
        assert graph_x.size() == (flattened_batch, hidden_dim)

        # This does
        # graph_x, graph_ids = self.graph_node_feature(
        #     graph_x, out_degree, graph_ids, num_total_graphs
        # )
        graph_x = self.graph_node_feature(graph_x, out_degree, num_total_graphs)

        # assert graph_ids.shape == graph_x.shape[:1], (
        #     f"Expected same shape, got {graph_ids.shape=} and"
        #     f" {graph_x.shape[:1]=}"
        # )

        # account for padding while computing the representation

        for block in self.blocks:
            graph_x, bert_output, vit_output, bottle_neck = block(
                graph_x,
                num_total_graphs,
                graph_mask,
                rotary_pos,
                bert_output,
                vit_output,
                bottle_neck,
                image_padding_mask,
                text_attention_mask,
            )

        graph_x = self.final_graphormer_layer(
            graph_x,
            mask=graph_mask,
            spatial_pos=rotary_pos,
        )

        if self.graph_token_average:
            global_embedding = self.average_embeddings_by_index(
                graph_x, graph_ids
            )
        else:
            global_embedding = graph_x[:num_total_graphs, :]

        assert global_embedding.size() == (num_total_graphs, hidden_dim)
        node_embedding = graph_x[num_total_graphs:, :]
        return node_embedding, global_embedding
