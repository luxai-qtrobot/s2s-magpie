from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ParakeetTDTSTTHandlerArguments:
    """
    Arguments for the Parakeet TDT Speech-to-Text handler.

    Parakeet TDT 0.6B v3 is a 600M parameter multilingual ASR model from NVIDIA.
    - On MPS (Apple Silicon): Uses mlx-audio with mlx-community/parakeet-tdt-0.6b-v3
    - On CUDA/CPU: Uses nano-parakeet (pure PyTorch) with nvidia/parakeet-tdt-0.6b-v3
    """

    parakeet_tdt_model_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The Parakeet TDT model to use. Defaults to 'mlx-community/parakeet-tdt-0.6b-v3' "
            "for MPS or 'nvidia/parakeet-tdt-0.6b-v3' for CUDA/CPU."
        },
    )
    parakeet_tdt_model_revision: Optional[str] = field(
        default=None,
        metadata={
            "help": "Immutable Hugging Face revision for nano-parakeet. The bundled model lock is used for the production model when unset."
        },
    )
    parakeet_tdt_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Optional Hugging Face cache directory containing the Parakeet checkpoint."},
    )
    parakeet_tdt_local_files_only: bool = field(
        default=True,
        metadata={"help": "Require nano-parakeet to use a pre-provisioned checkpoint. Default is True."},
    )
    parakeet_tdt_device: str = field(
        default="auto",
        metadata={
            "help": "Device to run the model on. 'auto' will use MPS on macOS and CUDA otherwise. "
            "Options: 'auto', 'cuda', 'mps', 'cpu'. Default is 'auto'."
        },
    )
    parakeet_tdt_compute_type: str = field(
        default="float16",
        metadata={"help": "Compute type for the model. Options: 'float16', 'float32'. Default is 'float16'."},
    )
    parakeet_tdt_language: Optional[str] = field(
        default=None,
        metadata={
            "help": "Target language code for transcription. If not specified, the model will "
            "auto-detect the language. Supports 25 European languages."
        },
    )
