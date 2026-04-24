"""PyRIT PromptTarget wrapper for Tinker's native sampling API.

Allows PyRIT attack orchestrators (like ManyShotJailbreakAttack) to
use Tinker-served models — both base and trained (LoRA) checkpoints —
without going through Tinker's unstable OpenAI-compatible endpoint.

Usage:
    from scripts.msj.tinker_target import TinkerTarget

    target = TinkerTarget(
        hf_model_id="meta-llama/Llama-3.1-8B-Instruct",
        sampler_path=None,  # or "tinker://.../.../sampler_weights/final"
    )
"""

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Optional

import tinker
from tinker_cookbook import renderers as r
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.tokenizer_utils import get_tokenizer

from pyrit.models import (
    ChatMessageRole,
    Message,
    MessagePiece,
    construct_response_from_request,
)
from pyrit.prompt_target import PromptChatTarget


class TinkerTarget(PromptChatTarget):
    """PyRIT PromptTarget that uses Tinker's native sampling API."""

    def __init__(
        self,
        hf_model_id: str,
        sampler_path: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        max_requests_per_minute: int = 30,
    ):
        super().__init__(max_requests_per_minute=max_requests_per_minute)

        self._hf_model_id = hf_model_id
        self._sampler_path = sampler_path
        self._max_tokens = max_tokens
        self._temperature = temperature

        # Initialize Tinker client and renderer
        self._client = tinker.ServiceClient()
        if sampler_path:
            self._sampling_client = self._client.create_sampling_client(
                model_path=sampler_path
            )
        else:
            self._sampling_client = self._client.create_sampling_client(
                base_model=hf_model_id
            )

        self._tokenizer = get_tokenizer(hf_model_id)
        try:
            renderer_name = get_recommended_renderer_name(hf_model_id)
        except (KeyError, ValueError):
            raise RuntimeError(f"Model '{hf_model_id}' not recognized by tinker_cookbook.")
        self._renderer = r.get_renderer(renderer_name, self._tokenizer)
        self._stop_sequences = self._renderer.get_stop_sequences()

        self._sampling_params = tinker.types.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            stop=self._stop_sequences,
        )

    @staticmethod
    def _expand_msj_turns(conversation: list[dict]) -> list[dict]:
        """Expand MSJ-style single-message prompts into multi-turn conversation.

        PyRIT's ManyShotJailbreakAttack constructs prompts like:
            User message: "You are a helpful assistant...
                User: <q1> Assistant: <a1>
                User: <q2> Assistant: <a2>
                ...
                User: <actual_objective>"

        This method splits the "User:"/"Assistant:" markers into actual
        conversation turns so the chat template creates real [INST]/[/INST]
        boundaries, which is critical for the in-context learning effect.
        """
        import re

        expanded = []
        for msg in conversation:
            if msg["role"] != "user":
                expanded.append(msg)
                continue

            content = msg["content"]

            # Check if this looks like an MSJ prompt (contains "User:" and "Assistant:" markers)
            if "User:" not in content or "Assistant:" not in content:
                expanded.append(msg)
                continue

            # Extract system-like preamble (before first "User:")
            first_user_idx = content.find("User:")
            preamble = content[:first_user_idx].strip()
            if preamble:
                expanded.append({"role": "system", "content": preamble})

            # Split on "User:" and "Assistant:" markers
            # Pattern: "User: <text> Assistant: <text>" repeating
            remainder = content[first_user_idx:]
            # Split into turns using regex
            turns = re.split(r'(?:^|\s)(User:|Assistant:)\s*', remainder)
            # turns[0] is empty or whitespace, then alternating marker/content
            current_role = None
            current_content = []

            for part in turns:
                part = part.strip()
                if not part:
                    continue
                if part == "User:":
                    if current_role and current_content:
                        expanded.append({
                            "role": current_role,
                            "content": " ".join(current_content).strip(),
                        })
                    current_role = "user"
                    current_content = []
                elif part == "Assistant:":
                    if current_role and current_content:
                        expanded.append({
                            "role": current_role,
                            "content": " ".join(current_content).strip(),
                        })
                    current_role = "assistant"
                    current_content = []
                else:
                    current_content.append(part)

            # Don't forget the last turn (the actual objective)
            if current_role and current_content:
                expanded.append({
                    "role": current_role,
                    "content": " ".join(current_content).strip(),
                })

        return expanded

    def _get_model_identifier(self) -> str:
        if self._sampler_path:
            return f"tinker:{self._sampler_path}"
        return f"tinker:{self._hf_model_id}"

    async def send_prompt_async(self, *, message: Message) -> list[Message]:
        """Send a prompt to Tinker and return the response as Message(s).

        Converts PyRIT's Message format to Tinker's chat format, calls
        the sampling API, and converts the response back.
        """
        # Extract conversation from message pieces
        raw_conversation = []
        for piece in Message.flatten_to_message_pieces([message]):
            role = piece.role
            content = piece.converted_value or piece.original_value
            if role == "system":
                raw_conversation.append({"role": "system", "content": content})
            elif role == "user":
                raw_conversation.append({"role": "user", "content": content})
            elif role in ("assistant", "simulated_assistant"):
                raw_conversation.append({"role": "assistant", "content": content})

        # MSJ sends the entire prompt as one user message with "User:"/"Assistant:"
        # markers. Split into actual multi-turn conversation so the chat template
        # creates real assistant turns (critical for in-context learning to work).
        conversation = self._expand_msj_turns(raw_conversation)

        # Build the prompt using Tinker's renderer
        model_input = self._renderer.build_generation_prompt(conversation)

        # Call Tinker sampling API (synchronous, wrap in executor)
        loop = asyncio.get_event_loop()
        future = self._sampling_client.sample(
            prompt=model_input,
            num_samples=1,
            sampling_params=self._sampling_params,
        )

        # future.result() is blocking — run in executor
        result = await loop.run_in_executor(None, future.result)

        # Parse response
        seq = result.sequences[0]
        parsed_msg, _ = self._renderer.parse_response(seq.tokens)
        completion_text = r.get_text_content(parsed_msg)

        # Build PyRIT response Message from the last request piece
        pieces = Message.flatten_to_message_pieces([message])
        # Find the last user piece to use as the request reference
        last_user_piece = [p for p in pieces if p.role == "user"][-1] if pieces else pieces[-1]

        response_message = construct_response_from_request(
            request=last_user_piece,
            response_text_pieces=[completion_text],
        )

        return [response_message]

    def is_json_response_supported(self) -> bool:
        return False

    def __str__(self) -> str:
        if self._sampler_path:
            return f"TinkerTarget(trained={self._sampler_path})"
        return f"TinkerTarget(base={self._hf_model_id})"
