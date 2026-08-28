from __future__ import annotations

import logging
from threading import Event
from time import perf_counter
from typing import Any, Iterator

import numpy as np
from rich.console import Console

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.handler_types import TTSIn, TTSOut
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker

logger = logging.getLogger(__name__)
console = Console()


class PocketTTSHandler(BaseHandler[TTSIn, TTSOut]):
    """
    Handler for Pocket TTS model from Kyutai Labs.
    Supports streaming audio generation with voice cloning.
    """

    def setup(
        self,
        should_listen: Event,
        device: str = "cpu",
        voice: str = "alba",  # Default voice from catalog
        sample_rate: int = 16000,  # Match the pipeline's native audio output rate.
        blocksize: int = 512,
        max_tokens: int = 50,
        gen_kwargs: dict[str, Any] | None = None,  # For compatibility with pipeline, not used
        cancel_scope: CancelScope | None = None,
        speculative_turns: SpeculativeTurnTracker | None = None,
    ) -> None:
        """
        Initialize Pocket TTS handler.

        Args:
            should_listen: Event to control when to start listening again
            device: Device to run model on ('cpu', 'cuda', 'mps')
            voice: Voice to use. Can be:
                - A preset name: 'alba', 'marius', 'javert', 'jean', 'fantine', 'cosette', 'eponine', 'azelma'
                - A local audio file path
                - A Hugging Face path like "hf://kyutai/tts-voices/..."
            sample_rate: Output sample rate (pocket-tts generates at 24kHz and will be resampled to this rate). Default 16kHz matches the pipeline's audio output.
            blocksize: Size of audio blocks to yield
            max_tokens: Maximum tokens to generate
        """
        self.should_listen = should_listen
        self.cancel_scope = cancel_scope
        self.speculative_turns = speculative_turns
        self.device = device
        self.voice = voice
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.max_tokens = max_tokens

        # Suppress verbose logging from pocket_tts library
        logging.getLogger("pocket_tts").setLevel(logging.WARNING)
        logging.getLogger("pocket_tts.models.tts_model").setLevel(logging.WARNING)
        logging.getLogger("pocket_tts.utils.utils").setLevel(logging.WARNING)

        # Import and load model
        from pocket_tts import TTSModel

        logger.info("Loading Pocket TTS model")
        self.model = TTSModel.load_model()

        # Move model to specified device
        if device == "cuda":
            self.model = self.model.cuda()
        elif device == "mps":
            self.model = self.model.to("mps")
        elif device != "cpu":
            self.model = self.model.to(device)

        logger.info(f"Pocket TTS model moved to {device}")

        # Load voice state
        logger.info(f"Loading voice: {voice}")
        self.voice_state = self.model.get_state_for_audio_prompt(voice)

        logger.info(f"Pocket TTS model sample rate: {self.model.sample_rate}")

    @property
    def min_time_to_debug(self) -> float:
        """
        Override to suppress logging for individual audio chunks.
        Pocket TTS yields many small chunks (~10-20ms each), which would flood logs.
        Only log if a chunk takes unusually long (>100ms).
        """
        return 0.1  # 100ms threshold

    @staticmethod
    def _to_int16(audio: np.ndarray) -> np.ndarray:
        return np.clip(audio * 32768, -32768, 32767).astype(np.int16)

    def process(self, tts_input: TTSIn) -> Iterator[TTSOut]:
        speculative_turns = getattr(self, "speculative_turns", None)
        if isinstance(tts_input, EndOfResponse):
            if speculative_turns and not speculative_turns.is_latest_after_reopen_grace(
                tts_input.turn_id,
                tts_input.turn_revision,
            ):
                if tts_input.response_key is None:
                    return
                tts_input.cleanup_only = True
            yield AUDIO_RESPONSE_DONE
            return

        if speculative_turns and not speculative_turns.is_latest_after_reopen_grace(
            tts_input.turn_id,
            tts_input.turn_revision,
        ):
            logger.debug("Dropping stale TTS input for turn=%s rev=%s", tts_input.turn_id, tts_input.turn_revision)
            return
        if speculative_turns:
            speculative_turns.commit(tts_input.turn_id, tts_input.turn_revision)

        gen = self.cancel_scope.generation if self.cancel_scope else None
        language_code = tts_input.language_code
        text = tts_input.text
        logger.debug(f"Received language code: {language_code}")

        console.print(f"[green]ASSISTANT: {text}")

        # Generate audio stream
        logger.debug(f"Generating audio for: {text[:50]}...")

        pipeline_start = perf_counter()
        first_chunk = True

        needs_resampling = self.model.sample_rate != self.sample_rate
        if needs_resampling:
            import soxr

            resampler = soxr.ResampleStream(
                self.model.sample_rate,
                self.sample_rate,
                num_channels=1,
                dtype="float32",
            )
        else:
            resampler = None

        audio_buffer = np.empty(0, dtype=np.float32)

        for audio_chunk in self.model.generate_audio_stream(
            self.voice_state,
            text,
            max_tokens=self.max_tokens,
            copy_state=True,  # Don't modify the original voice state
        ):
            if gen is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(gen):
                logger.info("TTS generation cancelled (interruption)")
                return
            if first_chunk:
                logger.debug(f"Time to first audio: {perf_counter() - pipeline_start:.3f}s")
                first_chunk = False

            audio_np = np.ascontiguousarray(audio_chunk.cpu().numpy(), dtype=np.float32).reshape(-1)
            if resampler is not None:
                audio_np = resampler.resample_chunk(audio_np, last=False)

            if audio_np.size:
                audio_buffer = np.concatenate((audio_buffer, audio_np))

            while audio_buffer.size >= self.blocksize:
                yield self._to_int16(audio_buffer[: self.blocksize])
                audio_buffer = audio_buffer[self.blocksize :]

        if resampler is not None:
            resampled_tail = resampler.resample_chunk(np.empty(0, dtype=np.float32), last=True)
            if resampled_tail.size:
                audio_buffer = np.concatenate((audio_buffer, resampled_tail))

        while audio_buffer.size >= self.blocksize:
            yield self._to_int16(audio_buffer[: self.blocksize])
            audio_buffer = audio_buffer[self.blocksize :]

        if audio_buffer.size:
            final_chunk = self._to_int16(audio_buffer)
            yield np.pad(final_chunk, (0, self.blocksize - final_chunk.size))
