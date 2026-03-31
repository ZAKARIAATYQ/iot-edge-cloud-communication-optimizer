# Real Multi-Sensor IoT-Edge-Cloud Predictive Simulation

## Overview

This module extends the multi-sensor predictive filtering framework by replacing **synthetic noise-based sensors** with **real geographically distinct datasets** from 5 Moroccan cities.

Each city acts as an independent IoT temperature sensor, providing naturally diverse signals that differ in altitude, climate zone, and local geography.

## Sensor Network

| Sensor ID | City              | Region             | Characteristics                |
|-----------|-------------------|--------------------|--------------------------------|
| 0         | Azilal            | High Atlas         | Mountain climate, high altitude |
| 1         | Beni Mellal       | Tadla-Azilal       | Semi-arid, plains              |
| 2         | Fquih Ben Salah   | Béni Mellal-Khénifra| Agricultural plains            |
| 3         | Khenifra          | Middle Atlas       | Continental, forested          |
| 4         | Marrakech         | Marrakech-Safi     | Hot semi-arid, urban heat      |

**Data source:** ERA5-Land hourly 2m air temperature (`t2m`), converted from Kelvin to Celsius.

## Difference vs. Synthetic Sensors

| Aspect                  | Synthetic (existing)           | Real (this module)             |
|-------------------------|--------------------------------|--------------------------------|
| Signal diversity        | Gaussian noise around one base | Naturally different climates   |
| Correlation structure   | i.i.d. noise                   | Spatially correlated weather   |
| Variability             | Controlled σ parameter         | Natural, city-dependent        |
| Number of sensors       | 10, 50, 100                    | 5 (one per city)               |
| Scientific value        | Scalability stress test        | Real-world validation          |

## Impact on Results

Real sensors introduce:
- **Higher inter-sensor variability** due to genuine climate differences (e.g., mountain vs. plains).
- **Spatial correlation** — nearby cities share weather patterns, affecting transmission decisions.
- **Non-uniform reduction rates** — cities with smoother temperature profiles transmit less.
- **Realistic critical events** — actual temperature spikes from real meteorological events.

## Experimental Setup

- **Model:** Pre-trained GRU (window=72, hidden=64), identical for Edge and Cloud (Dual Prediction Scheme).
- **Scaler:** Single global MinMaxScaler fitted on ALL sensor data combined.
- **Scenarios:** ε = 0.5°C and ε = 1.0°C thresholds.
- **Critical event threshold:** |ΔT| > 2°C between consecutive time steps.

## How to Run

```bash
cd simulation
python -m multi_sensor_real.simulate_multi_sensor_real
```

## Output Structure

```
multi_sensor_real/results/sensors_5/
├── summary.csv
├── epsilon_0_5/
│   ├── detailed_metrics.txt
│   ├── per_sensor_errors.csv
│   ├── per_sensor_metrics.csv
│   ├── critical_events.txt
│   ├── sample_sensor_reconstructions.png
│   ├── error_distribution_across_sensors.png
│   ├── validation_edge_vs_cloud.png
│   └── transmission_per_city.png
└── epsilon_1_0/
    └── (same structure)
```

## Metrics Computed

- **Global:** MAE, RMSE, R²
- **Per-sensor:** MAE, RMSE, R², transmission count, per-city reduction %
- **Communication:** Total transmissions, reduction percentage
- **Robustness:** Worst-case MAE, std deviation across sensors
- **Safety:** Critical events detected, missed critical events, detection rate
