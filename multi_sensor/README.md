# Multi-Sensor IoT-Edge-Cloud Predictive Simulation

This directory expands the single-sensor GRU-based predictive filtering framework into a highly scalable, multi-sensor scenario reflecting realistic edge environments.

## Architecture and Roles

- **Edge Node:** Acts as the local aggregator and processor for `N` connected sensors. It maintains an independent state (prediction history, sliding window) for each sensor.
- **Cloud Interface:** The remote destination for data.
- **Dual Prediction Scheme (DPS):** To minimize communication without drifting, both the Edge and the Cloud run the *exact same* GRU model structure with identical weights. 
  - They both utilize the *reconstructed history* as their input window.
  - Because their inputs and internal models are strictly identical, the Cloud perfectly predicts what the Edge predicts.
  - The Edge can thus evaluate the potential error (`|Real - Predicted|`) on behalf of the Cloud, safely omitting transmission if the error is below proper thresholds (`epsilon`).

## Implementation Features

- **Batched Inference:** Predicts missing values for `[10, 50, 100]` sensors concurrently in single tensor operations, simulating robust edge hardware vector processing.
- **No Shared State:** Each sensor is entirely isolated within the matrix computations.
- **Validation:** Independent generation of evaluation metrics ensuring tracking of max deviation (worst-case), per-sensor variances, execution time limits, and communication cost reductions.

## Trade-offs Analysed

- **Epsilon Threshold Setup vs. Communication Need:** A higher variance target (e.g. 1.0 vs 0.5) significantly dampens network requirements at the expense of local granularity loss at the Cloud layer.
- **Computational Cost Insight:** Moving validation workload locally drastically diminishes data ingress sizes, while the inference overhead on Edge grows linearly with the number of sensors. 

## Reproducibility

Run the specific evaluation script directly. No alterations made to the prior system ensure isolated behavior verification:
```bash
python simulate_multi_sensor.py
```
Output results will segregate into dynamically generated `results/sensors_X/` directories containing summaries, individual scenario data, and respective performance visualizations.
