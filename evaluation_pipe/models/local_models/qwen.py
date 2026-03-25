"""Qwen3.5 VLM wrappers for shape-bias evaluation."""

from __future__ import annotations

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from ..base import BaseVLM, ModelResponse
from .. import register_model


class _Qwen35Base(BaseVLM):
    """Shared loading/inference logic for Qwen3.5 vision-language models."""

    _default_model_id: str  # set by subclasses

    def __init__(
        self,
        model_id: str | None = None,
        device: str = "cuda",
        **kwargs,
    ) -> None:
        model_id = model_id or self._default_model_id
        self._model_id = model_id
        self._device = device
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
            **kwargs,
        )
        self._model.eval()

    @property
    def name(self) -> str:
        return self._model_id

    def generate(
        self,
        images: list[Image.Image],
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
    ) -> ModelResponse:
        # Build multi-image chat message
        content: list[dict] = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        # Two-step processing: template → processor
        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": False},
        )
        inputs = self._processor(
            text=[text],
            images=images,
            return_tensors="pt",
        ).to(self._model.device)

        input_len = inputs["input_ids"].shape[1]

        gen_kwargs: dict = dict(max_new_tokens=max_new_tokens)
        if temperature == 0.0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature

        def _run():
            with torch.inference_mode():
                return self._model.generate(**inputs, **gen_kwargs)

        output_ids, elapsed = self._timed_generate(_run)

        new_ids = output_ids[:, input_len:]
        raw_text = self._processor.batch_decode(new_ids, skip_special_tokens=True)[0]
        num_tokens = new_ids.shape[1]

        return ModelResponse(
            raw_text=raw_text.strip(),
            generation_time_s=elapsed,
            model_name=self.name,
            num_tokens_generated=num_tokens,
        )

    def unload(self) -> None:
        del self._model
        del self._processor
        torch.cuda.empty_cache()


@register_model("qwen3.5-0.8b")
class Qwen35_08B(_Qwen35Base):
    """Qwen3.5-VL 0.8B wrapper."""

    _default_model_id = "Qwen/Qwen3.5-VL-0.8B"


@register_model("qwen3.5-4b")
class Qwen35_4B(_Qwen35Base):
    """Qwen3.5-VL 4B wrapper."""

    _default_model_id = "Qwen/Qwen3.5-VL-4B"
