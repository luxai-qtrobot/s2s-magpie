# luxai-s2s-magpie

MAGPIE-native deployment of the Hugging Face speech-to-speech runtime for
QTrobot. The distribution name and command are `luxai-s2s-magpie`.

The project retains S2S's VAD/Smart Turn, STT, LLM, TTS, session state,
response ordering, tool-event semantics, and generation-aware cancellation.
It replaces the FastAPI, OpenAI Realtime WebSocket, and WebRTC serving path
with four native MAGPIE streams.

```text
 QTrobot / agent                         luxai-s2s-magpie

 microphone -- AudioFrameRaw ---------> VAD -> STT -> LLM -> TTS
                /s2s/audio/input                    |
                                                    |
 speaker   <-- S2SAudioFrame -----------------------+
                /s2s/audio/output

 client events -- DictFrame ----------> SessionRuntime
                  /s2s/events/input       - session state
                                            - response ordering
 server events <-- DictFrame ------------ - cancellation
                  /s2s/events/output       - stale-output filtering
                                            - tool event flow
```

Audio is mono PCM16 at 16 kHz. It is never base64-encoded. One MAGPIE GID is
used for each assistant audio response, followed by an empty frame carrying
the same GID. Phase 1 serves one active robot session at a time. The first
client event opens a clean session; the transport-level
`magpie.session.close` returns a correlated `magpie.session.closing`
acknowledgement immediately, then `magpie.session.closed` after the pipeline
drains. Another client can start without reloading the model handlers.

## Upstream source

The required S2S core is vendored under `src/speech_to_speech` at the exact
revision recorded in [UPSTREAM.md](UPSTREAM.md). This keeps the deployment
self-contained and lets LuxAI remove unused server transports and dependencies
while retaining clear provenance for future upstream updates.

## Install

Create a dedicated environment, install the known-good CUDA-enabled
Torch/Torchaudio builds for the robot's JetPack and Python versions, then
install the service with its model-loader constraints. JetPack 7 does not
currently have a separate NVIDIA Jetson Torch distribution, so preserve the
exact CUDA build qualified on the target robot.

```bash
python3 -m venv .venv
source .venv/bin/activate
# Install the target-qualified Torch/Torchaudio builds here first.
uv pip install .
```

This distribution contains its pinned, modified `speech_to_speech` core and
replaces the upstream distribution at runtime. Do not install both packages in
the same environment; two distributions must not own that import namespace.

FastAPI, Uvicorn, WebSocket, WebRTC, aiortc, and local PortAudio dependencies
are not required. Qwen3-TTS uses its Torch backend by default; install
`.[qwen-ggml]` only when a compatible qwentts.cpp build is actually desired.

### Provision models before deployment

Runtime startup is deliberately offline: it will not download NLTK data,
load mutable GitHub branches, or resolve the current head of a Hugging Face
repository. Model repositories and revisions, NLTK checksums, and the three
model-loader package versions are recorded in
[`src/speech_to_speech/model_assets.lock.json`](src/speech_to_speech/model_assets.lock.json).

While the build/provisioning machine has network access, populate stable cache
directories and then verify them without network access:

```bash
python scripts/provision_assets.py \
  --hf-cache /opt/luxai/s2s-magpie/models/huggingface \
  --nltk-data /opt/luxai/s2s-magpie/models/nltk_data

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
HF_HUB_CACHE=/opt/luxai/s2s-magpie/models/huggingface \
NLTK_DATA=/opt/luxai/s2s-magpie/models/nltk_data \
python scripts/provision_assets.py --verify-only \
  --hf-cache /opt/luxai/s2s-magpie/models/huggingface \
  --nltk-data /opt/luxai/s2s-magpie/models/nltk_data
```

Use those same four environment variables when launching the service. The
bundled Paramify defaults also set `local_files_only: true`, so a missing or
wrong cache fails at startup with a provisioning error instead of silently
using the network. A custom model must be supplied as a local path or paired
with an immutable revision. Offline GGML Qwen deployments additionally need
explicit local talker and codec GGUF paths.

Because NVIDIA's Jetson Torch wheels are tied to JetPack and Python, a single
portable full lock would be misleading. After validating one robot image,
capture that environment alongside the image metadata:

```bash
uv pip freeze --strict --exclude-editable > requirements-jetson.lock.txt
```

## Configuration and run

[`config/config.yaml`](config/config.yaml) is the single Paramify schema for
both the native S2S modules (VAD, STT, LLM, and TTS) and the MAGPIE service.
The included defaults match the tested Jetson setup: CUDA Parakeet TDT,
llama.cpp Chat Completions, and the Torch Qwen3-TTS 0.6B CustomVoice model
with `Ono_Anna`.

Run with those defaults, or pass a different Paramify file as the first
argument:

```bash
luxai-s2s-magpie
luxai-s2s-magpie /path/to/config.yaml
```

Paramify also exposes the configured CLI-scoped values, for example
`--zmq-port`, `--llm-model-name`, and `--tts-speaker`. There is no second S2S
argument parser: Paramify values are adapted directly into S2S's native
backend dataclasses. Fields omitted by the bundled schema keep their upstream
dataclass defaults. The bundled schema covers the tested Parakeet TDT,
Chat Completions, and Qwen3-TTS deployment; additional backend fields can be
declared in the corresponding Paramify group without adding another parser.

The service follows the standard LuxAI node convention. Zeroconf advertises
the node ID `luxai-s2s-magpie` and its single base RPC port. Calling the empty
RPC service returns the system descriptor from which clients discover every
stream. With the default base port `50960`, the layout is:

| Offset | Direction | Endpoint | Topic / service | Frame |
|---:|---|---|---|---|
| +0 | RPC | `tcp://*:50960` | system descriptor, `/s2s/status` | RPC |
| +1 | into service | `tcp://*:50961` | `/s2s/audio/input` | `AudioFrameRaw` |
| +2 | out of service | `tcp://*:50962` | `/s2s/audio/output` | `S2SAudioFrame` |
| +3 | into service | `tcp://*:50963` | `/s2s/events/input` | `DictFrame` |
| +4 | out of service | `tcp://*:50964` | `/s2s/events/output` | `DictFrame` |

Only the base port is configured; the four stream ports are derived from it.

The current QTrobot client is in `../qtrobot_s2s_agent`. It connects directly
through MAGPIE; no local Realtime API client or HTTP server is involved.

## Control and tool events

The event streams retain the useful OpenAI Realtime-shaped session semantics:

- client to runtime: `session.update`, `conversation.item.create`,
  `response.create`, and `response.cancel`;
- runtime to client: speech start/stop, live/final transcription, response
  lifecycle, assistant transcript, function-call, and error events.

Microphone PCM travels only on the audio stream. An
`input_audio_buffer.append` event is rejected. Tools remain a client/agent
responsibility: the runtime emits the function call and accepts the resulting
`function_call_output` through `conversation.item.create`.
