"""Modules related to the primary DiscussionTransformer model.

This model relates to the MDT model as presented in

Hebert, L., Sahu, G., Guo, Y., Sreenivas, N. K., Golab, L., & Cohen, R. (2024).
Multi-Modal Discussion Transformer: Integrating Text, Images and Graph
Transformers to Detect Hate Speech on Social Media.
Proceedings of the AAAI Conference on Artificial Intelligence,
38(20), 22096-22104. https://doi.org/10.1609/aaai.v38i20.30213
"""

from typing import Optional, Tuple

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

from components.graph_fusion_layer import GraphFusionStack
from components.v1.custom_attn import MultiheadAttention
from components.v1.graph_encoder_layer import GraphEncoderStack
from components.v2.feature_layers import GraphAttnBias
from components.v2.feature_layers import GraphNodeFeature
from utils.pylogger import RankedLogger

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
    if isinstance(module, MultiheadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)


class DiscussionTransformer(nn.Module):
    """
    Primary model class to create discussion and comment embeddings.
    """

    embedding_dim: int
    graph_node_feature: GraphNodeFeature
    graph_attn_bias: GraphAttnBias
    embed_scale: float
    emb_layer_norm: Optional[nn.LayerNorm]
    vit_model: ViTModel
    vit_pooler: ViTPooler
    text_model: BertModel
    text_pooler: BertPooler
    fusion_layers: nn.ModuleList
    graphormer_layers: nn.ModuleList
    num_bottle_neck: int
    bottle_neck: nn.Embedding

    def __init__(
        self,
        graph_node_feature: GraphNodeFeature,
        graph_attn_bias: GraphAttnBias,
        graph_stack_config: DictConfig,
        vit_model_config: DictConfig,
        text_model_config: DictConfig,
        num_bottle_neck: int,
        num_fusion_stack: int = 1,
        fusion_stack_size: int = 1,
        embedding_dim: int = 768,
        encoder_normalize_before: bool = False,
        embed_scale: float = 1,
        num_graph_layers_to_freeze: int = 0,
        freeze_initial_encoders: bool = False,
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
        """
        super().__init__()

        # TODO(liamhebert): It might be worth having a pass-through for
        # no-fusion models, where we just use graph layers.
        assert num_fusion_stack > 0, "num_fusion_stack must be greater than 0"
        assert fusion_stack_size > 0, "fusion_stack_size must be greater than 0"

        self.embedding_dim = embedding_dim
        self.graph_node_feature = graph_node_feature
        self.graph_attn_bias = graph_attn_bias

        self.embed_scale = embed_scale

        self.emb_layer_norm: Optional[nn.LayerNorm] = None
        if encoder_normalize_before:
            self.emb_layer_norm = nn.LayerNorm(self.embedding_dim)

        total_fusion_layers = fusion_stack_size * num_fusion_stack
        (
            self.vit_model,
            vit_fusion_layers,
            self.vit_pooler,
        ) = self.build_vit_encoder(
            num_fusion_layers=total_fusion_layers,
            **vit_model_config,
        )  # type: ignore[misc]

        (
            self.text_model,
            text_fusion_layers,
            self.text_pooler,
        ) = self.build_bert_encoder(
            num_fusion_layers=total_fusion_layers,
            **text_model_config,
        )  # type: ignore[misc]

        assert len(text_fusion_layers) == len(vit_fusion_layers)
        assert (
            len(text_fusion_layers) == fusion_stack_size * num_fusion_stack
        ), f"Expected {total_fusion_layers=}, got {len(text_fusion_layers)=}"

        self.fusion_layers = self.build_fusion_layers(
            text_fusion_layers, vit_fusion_layers, fusion_stack_size
        )
        assert len(self.fusion_layers) == num_fusion_stack

        num_fusion_layers = len(self.fusion_layers)

        self.graphormer_layers = nn.ModuleList([])
        self.graphormer_layers.extend(
            [
                self.build_graphormer_graph_encoder_layer(
                    **graph_stack_config  # type: ignore[misc]
                )
                for _ in range(num_fusion_layers + 1)
            ]
        )

        self.num_bottle_neck = num_bottle_neck
        self.bottle_neck = nn.Embedding(num_bottle_neck, self.embedding_dim)

        def freeze_module_params(m: nn.Module):
            """Given a module, freeze all of its parameters.

            Args:
                m (nn.Module): Module to freeze
            """
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False

        def unfreeze_module_params(m: nn.Module):
            """Given a module, unfreeze all of its parameters.

            Args:
                m (nn.Module): Module to unfreeze
            """
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = True

        if freeze_initial_encoders:
            freeze_module_params(self.text_model)
            freeze_module_params(self.vit_model)
            unfreeze_module_params(self.text_pooler)
            unfreeze_module_params(self.vit_pooler)

        for layer in range(num_graph_layers_to_freeze):
            freeze_module_params(self.graphormer_layers[layer])

    def build_fusion_layers(
        self,
        text_fusion_layers: list[BertLayer],
        vit_fusion_layers: list[ViTLayer],
        fusion_stack_size: int,
    ) -> nn.ModuleList:
        """Fuses the lists of modality layers into a list of fusion layers.

        Here, we group the layers into groups of `fusion_stack_size` layers,
        where each layer will process the output of the previous layer. We use
        num_bottle_neck to assist the fusion layers to know where the bottleneck
        tokens are.

        NOTE: the number of text and vision layers must be equal and divisible
        by `fusion_stack_size`.

        Args:
            text_fusion_layers (list[BertLayer]): List of text layers to fuse.
            vit_fusion_layers (list[ViTLayer]): List of vision layers to fuse.
            fusion_stack_size (int): Number of consecutive fusion layers to
                stack.
            num_bottle_neck (int): Number of bottleneck tokens we will be using.

        Returns:
            nn.ModuleList[GraphFusionStack]: The list of stacked fusion layers.
        """

        # TODO(liamhebert): Consider using itertools.zip_longest to handle
        # uneven lengths
        text_fusion_layers = [
            text_fusion_layers[
                i * fusion_stack_size : (i + 1) * fusion_stack_size
            ]
            for i in range(
                (len(text_fusion_layers) + fusion_stack_size - 1)
                // fusion_stack_size
            )
        ]
        vit_fusion_layers = [
            vit_fusion_layers[
                i * fusion_stack_size : (i + 1) * fusion_stack_size
            ]
            for i in range(
                (len(vit_fusion_layers) + fusion_stack_size - 1)
                // fusion_stack_size
            )
        ]

        fusion_layers = nn.ModuleList(
            [
                GraphFusionStack(text_layer, vit_layer, use_projection=False)
                for text_layer, vit_layer in zip(
                    text_fusion_layers, vit_fusion_layers
                )
            ]
        )
        return fusion_layers

    def build_vit_encoder(
        self,
        vit_model_name: str,
        attention_dropout: float,
        activation_dropout: float,
        num_fusion_layers: int,
        test_config: ViTConfig | None = None,
    ) -> Tuple[ViTModel, list[ViTLayer], ViTPooler]:
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
            Tuple[ViTModel, list[ViTLayer], ViTPooler]: A tuple conisting of
                - The Vision Transformer model with the last `num_fusion_layers`
                    layers removed.
                - The removed layers, to be used for fusion.
                - The pooler layer of the model, used to get the final output.
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
                hidden_dropout_prob=activation_dropout,
                attention_probs_dropout_prob=attention_dropout,
            )

        if num_fusion_layers == 0:
            vit_other_layers = []
        else:
            num_fusion_layers = num_fusion_layers
            vit_other_layers = vit_model.encoder.layer[-num_fusion_layers:]
            vit_model.encoder.layer = vit_model.encoder.layer[
                :-num_fusion_layers
            ]

        return vit_model, vit_other_layers, vit_model.pooler

    def build_bert_encoder(
        self,
        bert_model_name: str,
        attention_dropout: float,
        activation_dropout: float,
        num_fusion_layers: int,
        test_config: BertConfig | None = None,
    ) -> Tuple[BertModel, list[BertLayer], BertPooler]:
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
                - The pooler layer of the model, used to get the final output.
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
                hidden_dropout_prob=activation_dropout,
                attention_probs_dropout_prob=attention_dropout,
            )

        if num_fusion_layers == 0:
            bert_other_layers = []
        else:
            num_fusion_layers = num_fusion_layers
            bert_other_layers = bert.encoder.layer[-num_fusion_layers:]
            bert.encoder.layer = bert.encoder.layer[:-num_fusion_layers]

        return bert, bert_other_layers, bert.pooler

    def build_graphormer_graph_encoder_layer(
        self,
        embedding_dim: int,
        ffn_embedding_dim: int,
        num_attention_heads: int,
        dropout: float,
        attention_dropout: float,
        activation_dropout: float,
        activation_fn: nn.Module,
        num_layers=1,
    ) -> GraphEncoderStack:
        """Builds a stack of consecutive Graphormer encoder layers.

        This class is useful to make it easier to configure consecutive layers.
        See GraphEncoderStack for more details.

        Args:
            embedding_dim (int): The dimension of the input embeddings.
            ffn_embedding_dim (int): The dimension of the feedforward network
                used in attention.
            num_attention_heads (int): The number of attention heads. Must
                divide embedding_dim.
            dropout (float): The input dropout probability.
            attention_dropout (float): The dropout probability for attention
                weights.
            activation_dropout (float): The dropout probability for the
                activation function.
            activation_fn (nn.Module): The activation function to use.
            num_layers (int, optional): The number of consecutive graphormer
                layers to stack. Defaults to 1.

        Returns:
            GraphEncoderStack: A network module that calls num_layers of
                GraphEncoderLayer consecutively.
        """
        return GraphEncoderStack(
            num_layers=num_layers,
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            activation_fn=activation_fn,
        )

    def forward(
        self,
        text_input_ids: torch.Tensor,
        text_token_type_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        image_inputs: torch.Tensor,
        image_padding_mask: torch.Tensor,
        graph_ids: torch.Tensor,
        spatial_pos: torch.Tensor,
        in_degree: torch.Tensor,
        out_degree: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The forward function of the Discussion Transformer model.

        Args:
            == Text inputs ==
            text_input_ids (torch.Tensor): batched tokenized text ids, with shape
                (batch_size * nodes, T)
            text_token_type_ids (torch.Tensor): batched token type ids, with shape
                (batch_size * nodes, N, T)
            text_attention_mask (torch.Tensor): batched text attention mask, with
                shape (batch_size * nodes, N, T), where 1 indicates a token that
                should be attended to and 0 indicates padding.

            == Image inputs ==
            image_inputs (torch.Tensor): batched and tokenized image features, with
                shape (batch_size * nodes, D)
            image_padding_mask (torch.Tensor): batched boolean tensor indicating
                whether a node has an image. Shape (batch_size * nodes).

            == Graph inputs ==
            graph_ids (torch.Tensor): Id of the graph each node belongs to,
                where padding nodes are assigned the value PADDING_GRAPH_ID, with
                shape (batch_size * nodes). This is used to mask out attention.
            spatial_pos (torch.Tensor): Matrix with shape
                (batch_size * nodes, batch_size * nodes, 2) indicating the
                number of up hops and down hops between each node in the graph.
            in_degree (torch.Tensor): batched in-degrees, corresponding to the
                in-degree of each node in the graph. Padded with 0s and shifted
                by 1. Shape (batch_size * nodes).
            out_degree (torch.Tensor): batched out-degrees, corresponding to the
                out-degree of each node in the graph. Padded with 0s and shifted
                by 1. Shape (batch_size * nodes).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Returns
                - node_embedding: The final node embeddings for each node in the
                    graph with shape (batch_size * nodes, C).
                - global_embedding: The final global embedding for the graph,
                    with shape (batch_size, C).
        """
        # Ideally, we keep a consistent shape and just flatten here, rather then
        # having to dynamically index here. Attention mask should handle this.
        bert_output = self.text_model(
            token_type_ids=text_token_type_ids,
            attention_mask=text_attention_mask,
            input_ids=text_input_ids,
        ).last_hidden_state

        flattened_batch, seq_len, hidden_dim = bert_output.size()

        if image_inputs is not None:
            # TODO(liamhebert): Add image padding mask
            vit_output = self.vit_model(image_inputs).last_hidden_state
        else:
            vit_output = None

        bottle_neck = self.bottle_neck.weight.repeat(flattened_batch, 1, 1)

        bert_output, vit_output, bottle_neck = self.fusion_layers[0](
            bert_output,
            vit_output,
            bottle_neck,
            text_attention_mask,
            image_padding_mask,
        )

        bottle_neck_nodes = bottle_neck[:, 0, :]
        assert bottle_neck_nodes.size() == (flattened_batch, hidden_dim)

        # compute padding mask. This is needed for multi-head attention
        graph_x = bottle_neck_nodes
        # Assumes that PADDING_GRAPH_ID is -1
        unique_graph_ids = torch.count_nonzero(graph_ids) + 1

        padding_mask_cls = torch.zeros(
            unique_graph_ids,
            1,
            device=padding_mask.device,
            dtype=padding_mask.dtype,
        )

        padding_mask = torch.cat((padding_mask_cls, padding_mask), dim=1)
        mask = torch.cat((padding_mask_cls, node_mask), dim=1)
        # B x (T+1) x 1

        graph_x = self.graph_node_feature(graph_x, in_degree, out_degree)

        # x: B x T x C

        attn_bias = self.graph_attn_bias(attn_bias, spatial_pos)

        graph_x = graph_x * self.embed_scale

        if self.emb_layer_norm is not None:
            graph_x = self.emb_layer_norm(graph_x)

        # account for padding while computing the representation

        for g_layer, f_layer in zip(
            self.graphormer_layers[:-1], self.fusion_layers[1:]
        ):
            graph_x, _ = g_layer(
                graph_x,
                self_attn_padding_mask=padding_mask,
                self_attn_mask=graph_attention_mask,
                self_attn_bias=attn_bias,
            )

            # extract bottle_neck tokens
            bottle_neck[:, 0, :] = graph_x[mask, :]

            bert_output, vit_output, bottle_neck = f_layer(
                bert_output,
                vit_output,
                bottle_neck,
                extended_attention_mask,
                image_indices,
            )

            graph_x[mask, :] = bottle_neck[:, 0, :]

        graph_x, _ = self.graphormer_layers[-1](
            graph_x,
            self_attn_padding_mask=padding_mask,
            self_attn_mask=graph_attention_mask,
            self_attn_bias=attn_bias,
        )

        global_embedding = graph_x[:, 0, :]
        node_embedding = graph_x[:, 1:, :]
        return node_embedding, global_embedding
