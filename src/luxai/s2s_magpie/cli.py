"""Service entry point for the MAGPIE-native S2S runtime."""

from __future__ import annotations

import asyncio
import signal
import sys
from pathlib import Path
from threading import Event
from typing import Any

from luxai.magpie.discovery import ZconfDiscovery
from luxai.magpie.utils import Logger
from paramify import Paramify

from speech_to_speech.s2s_pipeline import (
    ParsedArguments,
    build_arguments_from_parameters,
    build_pipeline_unit,
    prepare_all_args,
    setup_logger,
)
from speech_to_speech.utils.thread_manager import ThreadManager

from .host import MagpieSessionHost
from .protocol import (
    AUDIO_INPUT_PORT_OFFSET,
    AUDIO_OUTPUT_PORT_OFFSET,
    EVENT_INPUT_PORT_OFFSET,
    EVENT_OUTPUT_PORT_OFFSET,
    PIPELINE_SAMPLE_RATE,
)


def _default_config_path() -> Path:
    """Locate the configuration in a source checkout or installed package."""

    source_config = Path(__file__).resolve().parents[3] / "config" / "config.yaml"
    if source_config.is_file():
        return source_config

    installed_config = (
        Path(sys.prefix)
        / "share"
        / "luxai-s2s-magpie"
        / "config"
        / "config.yaml"
    )
    if installed_config.is_file():
        return installed_config

    raise FileNotFoundError(
        "Could not find luxai-s2s-magpie config.yaml; pass its path as the "
        "first positional argument."
    )


def _load_parameters() -> tuple[Path, Any]:
    """Load one Paramify schema for both MAGPIE and native S2S settings."""

    config_path = (
        Path(sys.argv[1]).expanduser()
        if len(sys.argv) > 1 and not sys.argv[1].startswith("-")
        else _default_config_path()
    )
    return config_path, Paramify(str(config_path)).parameters


def _prepare_pipeline(parameters: Any) -> ParsedArguments:
    Logger.set_level(str(parameters.log_level))
    # Untouched vendored S2S handlers still use Python logging internally.
    setup_logger(str(parameters.log_level))
    args = build_arguments_from_parameters(parameters)
    prepare_all_args(args)
    if args.vad_handler_kwargs.sample_rate != PIPELINE_SAMPLE_RATE:
        raise ValueError(
            "MAGPIE S2S currently requires a 16 kHz VAD/audio pipeline"
        )
    return args


def _advertise_discovery(parameters: Any) -> ZconfDiscovery:
    """Create and advertise Zeroconf outside the asyncio event-loop thread."""

    discovery = ZconfDiscovery()
    try:
        discovery.advertise_node(
            str(parameters.zmq.node_id),
            port=int(parameters.zmq.port),
            payload={"version": str(parameters.service_version)},
        )
    except BaseException:
        discovery.close()
        raise
    return discovery


async def _serve(
    args: ParsedArguments,
    parameters: Any,
    stop_event: Event,
) -> None:
    unit = build_pipeline_unit(
        index=0,
        stop_event=stop_event,
        module_kwargs=args.module_kwargs,
        vad_handler_kwargs=args.vad_handler_kwargs,
        stt_backend=args.stt_backend,
        llm_backend=args.llm_backend,
        tts_backend=args.tts_backend,
    )
    manager = ThreadManager(unit.handlers)
    host = MagpieSessionHost(unit, stop_event, parameters)
    discovery: ZconfDiscovery | None = None
    shutdown_requested = asyncio.Event()
    loop = asyncio.get_running_loop()

    for signal_number in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signal_number, shutdown_requested.set)
        except (NotImplementedError, RuntimeError):
            # The default Windows event loop lacks add_signal_handler.
            pass

    try:
        manager.start()
        await host.start()
        # Zeroconf's synchronous API raises EventLoopBlocked when invoked from
        # the asyncio loop that it needs to coordinate with internally.
        discovery = await asyncio.to_thread(_advertise_discovery, parameters)

        base_port = int(parameters.zmq.port)
        Logger.info(
            "LuxAI S2S MAGPIE is ready "
            f"(node={parameters.zmq.node_id}, RPC={base_port}, "
            f"audio={base_port + AUDIO_INPUT_PORT_OFFSET}/"
            f"{base_port + AUDIO_OUTPUT_PORT_OFFSET}, "
            f"events={base_port + EVENT_INPUT_PORT_OFFSET}/"
            f"{base_port + EVENT_OUTPUT_PORT_OFFSET})"
        )

        host_wait = asyncio.create_task(host.wait(), name="magpie-s2s-host-wait")
        shutdown_wait = asyncio.create_task(
            shutdown_requested.wait(),
            name="magpie-s2s-shutdown-wait",
        )
        done, pending = await asyncio.wait(
            {host_wait, shutdown_wait},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        if host_wait in done:
            host_wait.result()
    finally:
        Logger.info("Stopping LuxAI S2S MAGPIE...")
        if discovery is not None:
            try:
                await asyncio.to_thread(discovery.close)
            except Exception as exc:
                Logger.warning(f"Failed to close MAGPIE discovery cleanly: {exc}")
        host_stop = asyncio.create_task(host.stop(), name="magpie-s2s-host-stop")
        drain_timeout = float(parameters.session.shutdown_drain_timeout_seconds)
        forced_timeout = float(parameters.session.forced_shutdown_timeout_seconds)
        try:
            try:
                await asyncio.wait_for(
                    asyncio.shield(host_stop),
                    timeout=drain_timeout,
                )
            except asyncio.TimeoutError:
                Logger.warning(
                    f"S2S session did not drain within {drain_timeout:g} seconds; "
                    "forcing pipeline shutdown"
                )
                stop_event.set()
                try:
                    await asyncio.wait_for(
                        asyncio.shield(host_stop),
                        timeout=forced_timeout,
                    )
                except asyncio.TimeoutError:
                    Logger.error(
                        "S2S host did not stop after the forced shutdown signal"
                    )
        finally:
            stop_event.set()
            manager.stop(timeout=forced_timeout)
            if not host_stop.done():
                host_stop.cancel()
            await asyncio.gather(host_stop, return_exceptions=True)


def main() -> int:
    """Run one S2S pipeline configured entirely through Paramify."""

    try:
        config_path, parameters = _load_parameters()
        Logger.set_level(str(parameters.log_level))
        Logger.info("LuxAI S2S MAGPIE starting")
        Logger.info(f"  config        : {config_path}")
        Logger.info(f"  version       : {parameters.service_version}")
        Logger.info(f"  ZMQ base port : {parameters.zmq.port}")

        args = _prepare_pipeline(parameters)
        stop_event = Event()
        asyncio.run(_serve(args, parameters, stop_event))
    except KeyboardInterrupt:
        Logger.info("LuxAI S2S MAGPIE stopped.")
    except Exception as exc:
        Logger.error(f"LuxAI S2S MAGPIE failed: {type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
