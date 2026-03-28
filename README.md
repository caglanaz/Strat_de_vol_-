# Lecture Volatility Trading

## Setup

Run the following commands to install the required packages:

```bash
pip install -r requirements.txt
pip install -e .
```

## Final project notebook

The final submission notebook is:

- `Notebook.ipynb`

It implements a volatility-timing carry strategy with a UKF on a Heston-type state-space model, and compares two observation specifications:

- `heston=True`: `E[r_t | v_t] = mu * dt`
- `heston=False` (mean modified): `E[r_t | v_t] = (mu - 0.5 * v_t) * dt`

Main reproducibility settings used in the notebook:

- rolling calibration window: `window = 63`
- rolling MLE recalibration: `recalib_every = 5`

For plotting, the active model case is selected with:

- `active_case = "heston"` or `active_case = "heston_mean_modified"`

# Disclaimer

- This course on financial derivatives was developed specifically for the Master 272 program at Paris Dauphine University and is intended for educational purposes only.
- The content reflects my personal views and interpretations, and does not represent the views, positions, or opinions of my employer or any affiliated institutions.
- While every effort has been made to ensure the accuracy and relevance of the material, this course does not constitute financial advice, nor does it replace professional consultation where appropriate.
- This material is licensed under the Creative Commons Attribution-NoDerivatives 4.0 International License (CC BY-ND 4.0). You are free to share and redistribute the content in any medium or format, including for commercial purposes, provided that appropriate credit is given and no modifications or derivative works are made.

# License

This project is licensed under the terms of the CC BY-NC-SA 4.0 license.

# Citation

```bibtex
@misc{VolatilityTradingCourse2026,
  title={Volatility Trading Course: Lecture Materials and Code},
  author={Baptiste ZLOCH},
  year={2026},
  howpublished={url{https://github.com/BaptisteZloch/Volatility-Investment-Course}},
  note={Python implementation of volatility surface modeling, SABR, and SSVI calibration, option strategies comprehensive backtests, etc. }
}
```
