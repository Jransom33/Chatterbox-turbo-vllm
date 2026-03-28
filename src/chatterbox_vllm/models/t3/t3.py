from typing import Iterable, Mapping, Optional, Sequence, Union

import torch
import torch.nn as nn
import random
from transformers.feature_extraction_utils import BatchFeature

from vllm.config import VllmConfig, ModelConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.interfaces import MultiModalEmbeddings, SupportsMultiModal
from vllm.model_executor.models.interfaces_base import VllmModelForTextGeneration
from vllm.model_executor.models.gpt2 import GPT2Model
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalKwargs, MultiModalKwargsItem, MultiModalBatchedField
from vllm.multimodal.parse import MultiModalDataParser, ModalityDataItems
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    MultiModalDataDict,
    MultiModalDataItems,
    MultiModalFieldConfig,
    PromptUpdate,
    MultiModalInputs,
    PlaceholderRange,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors

from chatterbox_vllm.models.t3.modules.t3_config import T3Config
from .modules.cond_enc import T3Cond, T3CondEnc


PREFILL_COND_START_TOKEN = 695  # [PLACEHOLDER55]; Marks the first token of the conditionals
PREFILL_COND_END_TOKEN = 696  # [PLACEHOLDER56]; Marks the last token of the conditionals
PREFILL_END_TOKEN = 697  # [PLACEHOLDER57]; Marks the end of the prefill block. This corresponds to the start of speech token.

CONDITIONING_SIZE = 376  # 1 for speaker_emb, 0 for clap_emb, 375 for raw cond_prompt_speech_emb, 0 for emotion_adv

# HACK: We need to be able to distinguish between the prefill tokens and the decode tokens.
# We'll do this by offsetting the speech tokens (only within vLLM) so they don't overlap with the
# normal speech tokens. This way, any token < SPEECH_TOKEN_OFFSET is a prefill token, and any token
# >= SPEECH_TOKEN_OFFSET is a decode token. This will only affect the logits and the encoding logic.
# No effect on the hidden states or the actual GPT2 model itself.
SPEECH_TOKEN_OFFSET = 50277  # Must be > all GPT2 token IDs (0-50275) to avoid collision with text tokens


class T3ProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"conditionals": 1}


class T3MultiModalDummyInputsBuilder(BaseDummyInputsBuilder):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return "Hello, world!"

    def get_dummy_mm_data(self, seq_len: int, mm_counts: Mapping[str, int]) -> MultiModalDataDict:
        return { "conditionals": [torch.zeros(CONDITIONING_SIZE, 1024)] * mm_counts["conditionals"] }


class T3MultiModalDataParser(MultiModalDataParser):
    def parse_mm_data(self, mm_data: MultiModalDataDict) -> MultiModalDataItems:
        conditionals: Optional[torch.Tensor] = mm_data.get("conditionals", None)
        if conditionals is None:
            return MultiModalDataItems({})

        return MultiModalDataItems({
            "conditionals": ConditionalsEmbeddingItems(conditionals)
        })


class ConditionalsEmbeddingItems(ModalityDataItems[torch.Tensor, torch.Tensor]):
    def __init__(self, data: torch.Tensor) -> None:
        super().__init__(data, "conditionals")

    def get_count(self) -> int:
        return 1

    def get(self, index: int) -> torch.Tensor:
        assert index == 0, index
        return self.data

    def get_processor_data(self) -> Mapping[str, torch.Tensor]:
        return {}

    def get_passthrough_data(self) -> Mapping[str, torch.Tensor]:
        return {"conditionals": self.data}


def create_triangular_matrix(m, n):
    # Create row indices and column indices
    row_indices = torch.arange(m).unsqueeze(1)  # Shape: (m, 1)
    col_indices = torch.arange(n).unsqueeze(0)  # Shape: (1, n)

    # Create the triangular mask
    matrix = (col_indices <= row_indices).float()

    return matrix


class T3MultiModalProcessor(BaseMultiModalProcessor[T3ProcessingInfo]):
    def _get_data_parser(self) -> MultiModalDataParser:
        return T3MultiModalDataParser()

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            conditionals=MultiModalFieldConfig.batched("conditionals")
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        # Bypassed via `apply` method.
        return []

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        tokenizer = self.info.get_tokenizer()
        processed_outputs = tokenizer(prompt, return_tensors="pt")
        processed_outputs['conditionals'] = mm_data.get('conditionals', None)
        return processed_outputs

    def apply(
        self,
        prompt: Union[str, list[int]],
        mm_data: MultiModalDataDict,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Optional[Mapping[str, object]] = None,
        return_mm_hashes: bool = False,
    ) -> MultiModalInputs:
        """
        Process multi-modal inputs to be used in vLLM.

        The main steps are:

        1. Apply HF Processor on prompt text and multi-modal data together,
           outputting token IDs and processed tensors.
        2. Find and update sequences in the token IDs with placeholder tokens.
           (SKIPPED for T3 conditioning)
        3. Extract information about the placeholder tokens from the
           processed token IDs.
           (Stubbed for T3 conditioning)
        """
        mm_items = self._to_mm_items(mm_data)

        (
            prompt_ids,
            mm_kwargs,
            mm_hashes,
            is_update_applied,
        ) = self._apply_hf_processor(
            prompt,
            mm_items,
            hf_processor_mm_kwargs,
            tokenization_kwargs,

            # Skip prompt caching calculation for now
            return_mm_hashes=False,
        )

        # The final embedding will look like <| cond | text | speech |>
        #
        # For prompt IDs, we replace the input tokens that match the conditionals with a
        # sequence of tokens that won't normally appear in the text prompt, to help unbatch.
        final_prompt_ids = [
            # Conditionals (totaling CONDITIONING_SIZE tokens)
            PREFILL_COND_START_TOKEN,
            *([prompt_ids[0]] * (CONDITIONING_SIZE-2)),
            PREFILL_COND_END_TOKEN,

            # Text prompt
            *prompt_ids,

            # Start of speech token / End of prefill block
            PREFILL_END_TOKEN,
        ]

        # HACK: Because vLLM can split the prefill across multiple batches, we need some way to
        # remember the offset of each text token.
        # We extend the conditioning embedding to cover the full prompt, filling in
        # the conditioning portion with the original embeddings, and the text portion with a
        # triangular matrix of 1s which encodes the offset of each text token.
        conditionals = mm_data.get("conditionals", None)
        assert conditionals is not None and len(conditionals) > 0, "Conditionals are required for prefill"
        assert len(conditionals) == 1, "Only one conditional embedding is supported for prefill"
        assert conditionals[0].shape[0] == CONDITIONING_SIZE, "Conditionals must be CONDITIONING_SIZE tokens long"

        new_conditionals = torch.cat([
            # First CONDITIONING_SIZE embeddings are the original conditionals
            conditionals[0],

            # The positions of the text ids are a triangular matrix of 1s
            create_triangular_matrix(len(prompt_ids), conditionals[0].shape[1]).to(conditionals[0].device),

            # The start of speech token is a vector of 0s
            torch.zeros(1, conditionals[0].shape[1]).to(conditionals[0].device),
        ], dim=0)
        assert len(new_conditionals) == len(final_prompt_ids), "Number of new conditionals does not match number of prompt ids"

        new_mm_kwargs = MultiModalKwargs.from_items([
            MultiModalKwargsItem.from_elems(
                MultiModalBatchedField().build_elems(
                    modality="conditionals",
                    key="conditionals",
                    data=[new_conditionals],
                )
            )
        ])

        return MultiModalInputs(
            type="multimodal",
            prompt=prompt,
            prompt_token_ids=final_prompt_ids,
            mm_kwargs=new_mm_kwargs,
            mm_hashes={
                # Assign a random hash for now, because we're not actually hashing the multimodal data.
                "conditionals": [str(random.random())],
            },
            mm_placeholders={
                # HACK: Tell vLLM that the conditionals modify the entire prompt. This will cause our hacked embeddings
                #       to be injected into the entire prompt, rather than just the conditioning portion.
                "conditionals": [PlaceholderRange(offset=0, length=len(final_prompt_ids), is_embed=None)]
            },
        )


@MULTIMODAL_REGISTRY.register_processor(T3MultiModalProcessor,
                                        info=T3ProcessingInfo,
                                        dummy_inputs=T3MultiModalDummyInputsBuilder)
class T3VllmModel(nn.Module, VllmModelForTextGeneration, SupportsMultiModal):
    """Native vLLM implementation of the Chatterbox T3 Turbo"""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        self.vllm_config = vllm_config
        self.cfg: ModelConfig = vllm_config.model_config

        # Initialize GPT2 backbone
        self.tfmr = GPT2Model(vllm_config=vllm_config, prefix=prefix + ".tfmr")

        # Initialize custom components
        self.t3conf = T3Config()
        self.dim = self.t3conf.n_channels
        self.cond_enc = T3CondEnc(self.t3conf)
        self.text_emb = nn.Embedding(self.t3conf.text_tokens_dict_size, self.dim)
        self.speech_emb = nn.Embedding(self.t3conf.speech_tokens_dict_size, self.dim)

        # logit projection (with separate bias for GPT2 mode)
        self.speech_head = ParallelLMHead(
            num_embeddings=self.t3conf.speech_tokens_dict_size,
            embedding_dim=self.dim,
            padding_size=1,
            prefix=prefix + ".speech_head",
        )
        self.speech_head_bias = nn.Parameter(torch.zeros(self.t3conf.speech_tokens_dict_size))
        self.logits_processor = LogitsProcessor(self.t3conf.speech_tokens_dict_size)


    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded_params: set[str] = set()
        state_dicts = {}
        hf_gpt2_weights = {}
        for name, weight in weights:
            # Handle speech_head.bias separately -> speech_head_bias
            if name == "speech_head.bias":
                self.speech_head_bias.data.copy_(weight)
                loaded_params.add("speech_head_bias")  # Use the nn.Parameter name, not the checkpoint key
                continue

            # GPT2 weights need to be passed through vllm's load_weights rather than load_state_dict
            if name.startswith("tfmr."):
                subname = name[5:]
                hf_gpt2_weights[subname] = weight
                continue
            loaded_params.add(name)
            attr, subname = name.split('.', 1)
            state_dict = state_dicts.get(attr, {})
            state_dict[subname] = weight
            state_dicts[attr] = state_dict

        for attr, state_dict in state_dicts.items():
            if hasattr(self, attr):
                getattr(self, attr).load_state_dict(state_dict)

        gpt2_loaded_params = self.tfmr.load_weights(hf_gpt2_weights.items())
        loaded_params.update('tfmr.' + i for i in gpt2_loaded_params)

        # Delete unused GPT2 word token embedding (we use custom embeddings)
        if hasattr(self.tfmr, 'wte'):
            del self.tfmr.wte

        return loaded_params


    def get_multimodal_embeddings(self, **kwargs: object) -> Optional[MultiModalEmbeddings]:
        conditionals: Optional[list[list[T3Cond]]] = kwargs.get("conditionals", [])
        return [batch[0] for batch in conditionals]


    def split_prefill_decode(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: list[MultiModalEmbeddings],
    ) -> list[torch.Tensor, Optional[MultiModalEmbeddings]]:
        """
        vLLM combines the prefill and decode into a single input tensor. We need to split them back
        out, and match the decode parts with the corresponding multimodal embeddings.

        Because of the SPEECH_TOKEN_OFFSET, the prefill tokens will always be <SPEECH_TOKEN_OFFSET and
        and the decode tokens will always be >= SPEECH_TOKEN_OFFSET.

        Furthermore, the prefill always starts with PREFILL_COND_START_TOKEN, and
        ends with PREFILL_END_TOKEN. However, vLLM can split the prefill across multiple batches,
        so we won't always have the complete prefill block in a single batch - we might only have the
        beginning or the end of a block.

        We can see back-to-back prefill blocks, so we can't just look for continuous sequences of
        prefill tokens. This nuance is not relevant for decode tokens as their position does not matter.

        Returns a list of tuples, where the first element is the input IDs for the block,
        and the second element is the associated multimodal embedding if the block is a prefill part.
        If the block is a decode part, the second element is None.
        """

        if len(input_ids) == 0:
            return []

        remaining_multimodal_embeddings = torch.cat(multimodal_embeddings, dim=0)

        in_prefill_block = input_ids[0] < SPEECH_TOKEN_OFFSET

        output = []

        # Keep a buffer of current tokens
        buffer = []

        for input_id in input_ids:
            # Check if we've swapped between prefill and decode blocks, or if we've just hit the start of a new prefill block
            if (in_prefill_block != (input_id < SPEECH_TOKEN_OFFSET)) or (input_id == PREFILL_COND_START_TOKEN):
                if buffer:
                    if in_prefill_block:
                        mme, remaining_multimodal_embeddings = remaining_multimodal_embeddings\
                            .split([len(buffer), len(remaining_multimodal_embeddings) - len(buffer)], dim=0)
                        output.append((torch.tensor(buffer).to(input_ids.device), mme))
                    else:
                        output.append((torch.tensor(buffer).to(input_ids.device), None))

                buffer = []
                in_prefill_block = (input_id < SPEECH_TOKEN_OFFSET)

            # Add new token to buffer
            buffer.append(input_id)

        # Add any elements left in the buffer
        if buffer:
            if in_prefill_block:
                mme, remaining_multimodal_embeddings = remaining_multimodal_embeddings\
                    .split([len(buffer), len(remaining_multimodal_embeddings) - len(buffer)], dim=0)
                output.append((torch.tensor(buffer).to(input_ids.device), mme))
            else:
                output.append((torch.tensor(buffer).to(input_ids.device), None))

        return output


    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            # There's no multimodal embeddings, so we're decoding.
            # Remember to undo the offset we applied to the speech tokens.
            return self.speech_emb(input_ids - SPEECH_TOKEN_OFFSET)
        else:
            out = []
            for ids, multimodal_embedding in self.split_prefill_decode(input_ids, multimodal_embeddings):

                if multimodal_embedding is None:
                    # Decoding speech tokens - undo the offset
                    embeds = self.speech_emb(ids - SPEECH_TOKEN_OFFSET)
                    out.append(embeds)
                    continue

                # We're in the prefill stage, and need to wrangle the multimodal embeddings into the right format.
                # Embeddings are in the format of <| cond | text | speech |>
                #
                # Due to vLLM batching, we may only get the first half or the last half of the prefill block.
                # We handle each case separately.

                if ids[0] == PREFILL_COND_START_TOKEN and ids[-1] == PREFILL_END_TOKEN:
                    # We have the full prefill block.
                    # The first CONDITIONING_SIZE tokens are the cond portion. The remainder, except
                    # for the last token are the text portion. The last token is the start of speech token.
                    text_ids = ids[CONDITIONING_SIZE:-1]
                    text_emb = self.text_emb(text_ids)

                    start_of_speech_token = torch.tensor([self.t3conf.start_speech_token]).to(ids.device)
                    start_of_speech_emb = self.speech_emb(start_of_speech_token.unsqueeze(0))[0]

                    conditioning_emb = multimodal_embedding[0:CONDITIONING_SIZE]
                    final_embeds = torch.cat([conditioning_emb, text_emb, start_of_speech_emb], dim=0)
                    out.append(final_embeds)

                elif ids[0] == PREFILL_COND_START_TOKEN:
                    # We have the start of the prefill block but not the end.
                    text_ids = ids[CONDITIONING_SIZE:]
                    text_emb = self.text_emb(text_ids)

                    conditioning_emb = multimodal_embedding[0:min(CONDITIONING_SIZE, len(multimodal_embedding))]
                    final_embeds = torch.cat([conditioning_emb, text_emb], dim=0)
                    assert len(final_embeds) == len(ids), "Number of output elements does not match number of input elements"
                    out.append(final_embeds)

                elif ids[-1] == PREFILL_END_TOKEN:
                    # We have the end of the prefill block.
                    # Check if the end-of-conditioning token is present.
                    indices = torch.where(ids == PREFILL_COND_END_TOKEN)[0]
                    if len(indices) > 0:
                        # We have some conditioning + the full text input
                        text_ids = ids[indices[0]+1:-1]
                        text_emb = self.text_emb(text_ids)

                        start_of_speech_token = torch.tensor([self.t3conf.start_speech_token]).to(ids.device)
                        start_of_speech_emb = self.speech_emb(start_of_speech_token.unsqueeze(0))[0]

                        conditioning_emb = multimodal_embedding[:indices[0]+1]

                        final_embeds = torch.cat([conditioning_emb, text_emb, start_of_speech_emb], dim=0)
                        out.append(final_embeds)
                    else:
                        # No conditioning portion, may only have part of the text portion.
                        text_ids = ids[:-1]
                        text_emb = self.text_emb(text_ids)

                        start_of_speech_token = torch.tensor([self.t3conf.start_speech_token]).to(ids.device)
                        start_of_speech_emb = self.speech_emb(start_of_speech_token.unsqueeze(0))[0]

                        final_embeds = torch.cat([text_emb, start_of_speech_emb], dim=0)
                        assert len(final_embeds) == len(ids), "Number of output elements does not match number of input elements"
                        out.append(final_embeds)

                else:
                    # Middle chunk of a prefill block due to vllm V1 chunked prefill.
                    # The block starts mid-sequence (not at PREFILL_COND_START_TOKEN) and
                    # doesn't end at PREFILL_END_TOKEN. All tokens here are either late
                    # conditioning tokens or text tokens - use multimodal_embedding directly
                    # for conditioning portions; text tokens get their embeddings from text_emb.
                    if PREFILL_COND_END_TOKEN in ids:
                        # Has the end-of-conditioning marker: split into conditioning + text
                        end_idx = (ids == PREFILL_COND_END_TOKEN).nonzero(as_tuple=True)[0][0]
                        cond_part = multimodal_embedding[:end_idx + 1]
                        text_ids = ids[end_idx + 1:]
                        text_part = self.text_emb(text_ids)
                        final_embeds = torch.cat([cond_part, text_part], dim=0)
                    else:
                        # Pure conditioning or pure text middle chunk - use multimodal_embedding
                        final_embeds = multimodal_embedding
                    assert len(final_embeds) == len(ids), f"Embedding size mismatch: {len(final_embeds)} vs {len(ids)}"
                    out.append(final_embeds)

            return torch.cat(out, dim=0)


    def compute_logits(self, hidden_states: torch.Tensor, sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.speech_head, hidden_states, sampling_metadata)

        # Add bias (GPT2 mode has bias on speech_head)
        logits = logits + self.speech_head_bias

        # HACK: Offset the logits so the resulting speech token is +SPEECH_TOKEN_OFFSET from the normal speech tokens.
        #       We'll do this by adding SPEECH_TOKEN_OFFSET fake dimensions to the left of the logits.
        #       This is a hack to help us unbatch batched inputs.
        logits = torch.cat([
            torch.zeros(logits.shape[0], SPEECH_TOKEN_OFFSET).to(logits.device).fill_(float('-inf')),
            logits,
        ], dim=1)
        return logits


    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings(input_ids, [])

        hidden_states = self.tfmr(
            input_ids=None,
            position_ids=positions,
            intermediate_tensors=None,
            inputs_embeds=inputs_embeds,
        )

        return hidden_states

    def get_language_model(self) -> torch.nn.Module:
        return self.tfmr
