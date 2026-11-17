# GTA VI Twitch CCU forecast — diurnal-profile lognormal forecaster

Entry for make.fun competition `gta6-twitch-ccu-forecast-01`: forecast
Twitch viewers in the Grand Theft Auto VI category for hours 2–24 of
release day (2026-11-19 UTC) from the hour-1 print, as five quantiles
per hour. Pure Python 3.13 standard library, no network, deterministic.

## Model

Launch-day Twitch viewership has a strong diurnal shape: it rises as
regions unlock through the day (NZ first, then Europe, then the US),
peaks in the evening UTC, and falls overnight. Persistence (every hour
= hour 1) misses that shape entirely.

For each hour h = 2..24, the program fits, at run time, an OLS line
over the case's own comparable launch days:

```
log(y_h / y_1) = a_h + b_h * log(y_1)
```

The slope is strongly negative at later hours: launches whose hour 1
is already huge stay near flat, while launches whose hour 1 is tiny
explode as the day's unlocks roll in. Conditioning on the hour-1 level
is what beats persistence; a single median profile does not. The pure
regression is dragged around by the extreme tiny-hour-1 launches, so
the final prediction blends it 40/60 with the median log-ratio of
launches in the same hour-1 band (under 5k, 5k–50k, over 50k viewers).

- Median: `p1 * exp(0.4 * (a_h + b_h * log(p1)) + 0.6 * band_median_h)`,
  log-ratio clamped to [-3, 7].
- Spread: `sigma_h` = residual SD of that hour's regression, floored
  at 0.12 and capped at 0.80 log units; quantiles are
  `median * exp(z_tau * sigma_h)`.
- Fallbacks: fewer than 8 usable comparables → median log-ratio
  profile; no usable comparables → persistence with a widening spread.

## Validation

Leave-one-out cross-validation on the 37 usable comparable launch days
in the kit's practice cases, scored with the contest's pinball loss on
the log scale:

| model | mean loss | 90% coverage | typical miss |
|---|---|---|---|
| this model | **0.380** | 0.902 | ~96% |
| persistence baseline | 0.930 | 0.783 | ~83% |

The model beats persistence on 28 of 37 folds, and on the 27 folds
with hour 1 ≥ 20,000 viewers — the regime the practice cases and the
GTA VI final almost surely sit in — it scores 0.276 against
persistence's 0.321.
launches whose hour 1 was a few hundred viewers and which then exploded
by three orders of magnitude — inherently hard to pin down, which is
what the wide late-hour sigma prices in.

## Run

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in the input directory and writes `<id>.out` for
each: `{"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]}`.
