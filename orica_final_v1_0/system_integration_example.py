"""Minimal integration example for the complete EEG/neurofeedback pipeline.

Expected upstream stages:
    Receiver -> Ring Buffer -> Preprocessor -> Rank Manager -> ORICA

Expected downstream stages:
    IC artifact detector -> IC removal -> reconstruction/features/neurofeedback
"""

import numpy as np

from orica.api import create_final_orica, process_orica_block


class RealtimeORICAStage:
    """Thin stateful wrapper intended for the larger real-time system."""

    def __init__(
        self,
        n_channels: int,
        sfreq: float,
        block_size: int = 8,
        backend: str = "auto",
    ):
        self.block_size = int(block_size)
        self.model, self.info = create_final_orica(
            n_channels=n_channels,
            sfreq=sfreq,
            block_size=block_size,
            backend=backend,
        )

    def process(self, preprocessed_fullrank_block, diagnostics: bool = False):
        """Process one preprocessed, full-rank block.

        The validated real-EEG pipeline used microvolts. Keep the signal unit
        consistent across the complete runtime.
        """
        X = np.asarray(preprocessed_fullrank_block, dtype=np.float64)
        return process_orica_block(
            self.model,
            X,
            return_diagnostics=diagnostics,
            output_mode="updated",
        )

    def reset(self):
        self.model.reset()


# -------------------------------------------------------------------------
# Example wiring (replace get_next_block() with the real Ring Buffer output)
# -------------------------------------------------------------------------
if __name__ == "__main__":
    stage = RealtimeORICAStage(
        n_channels=8,
        sfreq=500.0,
        block_size=8,
        backend="auto",
    )

    # Example only: one block from the upstream preprocessor/rank manager.
    X_block = np.zeros((8, 8), dtype=np.float64)

    # Real EEG should not be all-zero; this is only to demonstrate the API.
    # In the live loop use:
    # Y_block = stage.process(get_next_block())
    print(stage.info)
