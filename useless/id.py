{
  "id": "filter_grid_0123",
  "filter_type": "chebyshev",
  "spec": {
    "fc": 1.2e9,
    "fs": 2.4e9,
    "ripple_db": 0.2,
    "La_target": 40,
    "R0": 50,
    "order": 5
  },
  "design": {
    "L": [ ... ],
    "C": [ ... ],
    "N": 5
  },
  "metrics": {
    "S11_max_dB": -15.2,
    "S21_passband_min_dB": -0.6,
    "S21_stopband_max_dB": -42.1,
    "passband_ripple_dB": 0.18
  },
  "artifacts": {
    "netlist": "...",
    "csv": "path/to/SP1.SP.csv",
    "ds": "path/to/design.ds"
  }
}