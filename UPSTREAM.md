# Upstream source

The speech pipeline in `src/speech_to_speech` is derived from Hugging Face
[`speech-to-speech`](https://github.com/huggingface/speech-to-speech).

- Upstream commit: `b21e54d6a845c54308bc62500b57f9a2b2aa93ac`
- Imported: 2026-08-19
- Upstream distribution version: `0.2.12`
- License: Apache-2.0

LuxAI changes keep the VAD, Smart Turn, STT, LLM, TTS, conversation,
response-ordering, and cancellation machinery while replacing the FastAPI,
WebSocket, and WebRTC serving path with native MAGPIE streams.

When updating, first import a pristine snapshot of the new upstream commit,
then reapply the LuxAI commits in order. Never update the source without also
changing the commit recorded here.

