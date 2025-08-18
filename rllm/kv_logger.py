import os, json, logging, uuid, time
from logging.handlers import RotatingFileHandler

# Environment variables for configuring the output location
_RUN_ID = os.getenv("KV_LOG_RUN_ID", uuid.uuid4().hex[:8])
_LOG_ROOT = os.path.abspath(os.getenv("KV_LOG_DIR", "./kv_stats"))
_LOG_DIR = os.path.join(_LOG_ROOT, _RUN_ID)

os.makedirs(_LOG_DIR, exist_ok=True)

_rank_loggers: dict[int, logging.Logger] = {}
_input_loggers: dict[int, logging.Logger] = {}


def _create_logger_for_rank(rank: int) -> logging.Logger:
    """Return a per-rank logger that appends ND-JSON lines to a file."""
    logger = logging.getLogger(f"kvstats.rank-{rank}")
    if logger.handlers:
        return logger

    path = os.path.join(_LOG_DIR, f"rank-{rank}.log")
    handler = RotatingFileHandler(path, maxBytes=0, backupCount=0)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def _get_logger(rank: int) -> logging.Logger:
    if rank not in _rank_loggers:
        _rank_loggers[rank] = _create_logger_for_rank(rank)
    return _rank_loggers[rank]

# ---------------------------------------------------------------------------
# Input logger (stores raw prompt / input strings) in separate directory to
# avoid mixing with the compact hash-only log.  File structure mirrors the
# KV log directory but lives under kv_inputs/.
# ---------------------------------------------------------------------------

_INPUT_DIR = os.path.join(_LOG_ROOT, "kv_inputs", _RUN_ID)
os.makedirs(_INPUT_DIR, exist_ok=True)


def _create_input_logger_for_rank(rank: int) -> logging.Logger:
    logger = logging.getLogger(f"inputs.rank-{rank}")
    if logger.handlers:
        return logger
    path = os.path.join(_INPUT_DIR, f"rank-{rank}.log")
    handler = RotatingFileHandler(path, maxBytes=0, backupCount=0)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def _get_input_logger(rank: int) -> logging.Logger:
    if rank not in _input_loggers:
        _input_loggers[rank] = _create_input_logger_for_rank(rank)
    return _input_loggers[rank]


def dump(rank: int, prefix_hash: str, hit: bool | None = None, global_step: int | None = None, 
         latency_ms: float | None = None, extra: dict | None = None) -> None:
    """Append one JSON record about a prefix-cache lookup.

    Parameters
    ----------
    rank : int
        GPU/data-parallel rank emitting the record.
    prefix_hash : str
        Hex MD5 (or any) hash of the prompt prefix tokens.
    hit : bool | None
        True if KV blocks were found, False if newly created, None if unknown.
    global_step : int | None
        Optional training step number.
    latency_ms : float | None
        Time taken for this request in milliseconds.
    extra : dict | None
        Additional key/value pairs to include.
    """
    record: dict[str, object] = {
        "ts": time.time(),
        "hash": prefix_hash,
    }
    if hit is not None:
        record["hit"] = hit
    if global_step is not None:
        record["step"] = global_step
    if latency_ms is not None:
        record["latency_ms"] = latency_ms
    if extra:
        record.update(extra)

    _get_logger(rank).info(json.dumps(record, separators=(",", ":")))


def dump_input(rank: int, prompt: str, global_step: int | None = None) -> None:
    """Write raw prompt string for debugging/analysis.

    Stored in kv_inputs/<run-id>/rank-<rank>.log as newline-delimited JSON with
    fields: ts, step, prompt_len, prompt.
    """
    try:
        rec = {
            "ts": time.time(),
            "prompt_len": len(prompt),
            "prompt": prompt,
        }
        if global_step is not None:
            rec["step"] = global_step
        _get_input_logger(rank).info(json.dumps(rec, separators=(",", ":")))
    except Exception:
        pass