"""Simple performance logger that writes one line per batch."""

import os
import time
import json
from pathlib import Path

class PerfLogger:
    def __init__(self, run_id=None):
        if run_id is None:
            # Use timestamp as run-id if none provided
            run_id = time.strftime("%Y%m%d_%H%M%S")
            
        log_dir = Path("perf_stats") / run_id
        os.makedirs(log_dir, exist_ok=True)
        
        self.files = {}  # rank -> file handle
        self.log_dir = log_dir
        
    def log_batch(self, rank: int, step: int, batch_size: int, latency_ms: float):
        """Write one line with batch timing."""
        if rank not in self.files:
            path = self.log_dir / f"rank-{rank}.tsv"
            if not path.exists():
                # Write header on first open
                with open(path, "w") as f:
                    f.write("timestamp\tstep\tbatch_size\tlatency_ms\n")
            self.files[rank] = open(path, "a")
            
        f = self.files[rank]
        f.write(f"{time.time()}\t{step}\t{batch_size}\t{latency_ms:.1f}\n")
        f.flush()  # Make sure it hits disk
        
    def close(self):
        """Close all log files."""
        for f in self.files.values():
            f.close()
        self.files.clear()

# Global instance
_logger = None

def init(run_id=None):
    """Initialize the global logger instance."""
    global _logger
    if _logger is not None:
        _logger.close()
    _logger = PerfLogger(run_id)
    return _logger

def log_batch(rank: int, step: int, batch_size: int, latency_ms: float):
    """Log one batch through the global logger."""
    if _logger is None:
        init()
    _logger.log_batch(rank, step, batch_size, latency_ms)

def close():
    """Close the global logger."""
    global _logger
    if _logger is not None:
        _logger.close()
        _logger = None