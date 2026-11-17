"""InferenceCarbon metering harness.

Modules:
  templates   -- load architecture and serving templates
  work        -- forward calculations: K(L), W, W_batch, memory feasibility
  feasibility -- turn cells.yaml into a run manifest with GPU-hours
  power       -- Zeus/NVML wrapper with counter qualification
  server      -- vLLM launcher with the fixed campaign settings
  loadgen     -- request generator: cache warming, batch control, event timing
  cell        -- run one cell end to end -> Parquet
  campaign    -- iterate the manifest; checkpoint; resume; push
"""

__version__ = "0.1.0"
