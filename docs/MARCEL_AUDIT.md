# Marcel audit: how our baseline differs from the published Marcel

Criterion 1 of the design freeze is "`ridge_rel` MAE below `MarcelBaseline` MAE". The whole
pre-registered claim therefore rests on what `MarcelBaseline` is, and it is not Tom Tango's
Marcel. It is a simplification of it, written in `src/mlb_outlook/models/marcel.py` and
`models/baselines.py`. This file is the inventory of the differences, written before the fold 3
run so that none of it can be produced afterwards as an explanation.

Two things to say plainly up front:

- **Some of the differences weaken our Marcel**, which makes criterion 1 easier to pass than
  "beats Marcel" sounds. The single-rate projection (difference 1) is the main one.
- **One of them favours our Marcel**: it is only ever asked about players who already took 100
  plate appearances in the prior season (difference 9), which is the population it is strongest
  on. Tango's Marcel has to project rookies and part-timers it knows nothing about.

Nothing here is a change. The freeze pins `MarcelBaseline` as it stands, and the right response
to this audit is a v2 baseline, not an edit to the one the criterion was registered against.

## What we implement

For a hitter with prior-season pairs `(pa_k, woba_k)` at lags k = 1, 2, 3, weights
`w = (5, 4, 3)`, league rate `lg`, regression constant `R = 1200`, and target-season age `a`:

```
                 sum_k w_k * pa_k * woba_k  +  R * lg
    forecast  =  ------------------------------------  *  age_factor(a)
                 sum_k w_k * pa_k          +  R

    age_factor(a) = 1 + (29 - a) * (0.006 if a < 29 else 0.003)
```

`lg` is the recency-weighted mean of the three lag seasons' league wOBA, weighted 5/4/3 and not
by plate appearances. A lag season the player did not play contributes nothing to either sum.

What that does, in numbers:

| Prior PA (lag 1/2/3) | Weighted PA | Share of the forecast that is league average | A .360 hitter projects to |
|---|---|---|---|
| 600 / 600 / 600 | 7200 | 14.3% | 0.3529 |
| 600 / 0 / 0 | 3000 | 28.6% | 0.3457 |
| 150 / 0 / 0 | 750 | 61.5% | 0.3292 |

| Age | Factor | A .320 line becomes |
|---|---|---|
| 22 | 1.0420 | 0.3334 |
| 26 | 1.0180 | 0.3258 |
| 29 | 1.0000 | 0.3200 |
| 32 | 0.9910 | 0.3171 |
| 35 | 0.9820 | 0.3142 |

## The differences

**1. One rate, not a line of components.** Tango's Marcel projects each batting event per plate
appearance -- singles, doubles, triples, home runs, walks, strikeouts, hit-by-pitch -- regresses
each with its own amount, and assembles the rate from the projected components. We project wOBA
directly as a single rate with a single regression constant. This is the difference that matters
most and it cuts against our Marcel: component-wise regression uses the fact that some events
stabilise much faster than others (strikeout rate in a few hundred PA, home-run rate far
slower), and collapsing them into one rate throws that away. **Our Marcel is therefore a weaker
forecaster than the published Marcel, and criterion 1 is correspondingly easier.** How much
easier is not quantified here, because building a component Marcel is a new baseline and the
freeze pins the baselines. It is the first item on the v2 list for this file.

**2. No playing-time projection.** Marcel projects plate appearances as well as rates. We do
not, because the population is defined by a 100-PA floor in the target season and nothing in the
evaluation weights a forecast by projected PA. Consequence: `MarcelBaseline` cannot produce a
full projected line, only a rate, and the PA-weighted metrics in `PROJECT_PLAN.md` weight by
*realized* PA rather than projected PA.

**3. The regression constant, and an ambiguity in the published figure.** We use `R = 1200`
against a denominator of `sum w_k * pa_k`, so for a full-time hitter 1200 of 8400 weighted PA --
14.3% -- is league average. Tango's archived description scales the weighted league line to
1200 PA, "two weights times 600 PA", which is the same shape. The ambiguity: the published
number is quoted both as 1200 on a weight-multiplied PA scale and as ~100 PA on a
weights-normalised scale (the weights sum to 12, and 1200 / 12 = 100), and third-party
reimplementations differ. The two readings are the same thing, but if the intended scale were
unweighted PA then our Marcel would be regressing roughly twelve times too little. We use the
weighted-scale reading, which is the one that makes a full-time hitter's forecast ~14% league
average -- the magnitude the literature on wOBA reliability supports.

**4. The aging adjustment is multiplicative, and the published form is unclear.** We multiply
the regressed rate by `1 + (29 - age) * slope`. The slopes 0.006 and 0.003 either side of 29 are
the widely quoted Marcel values, but whether they are meant multiplicatively or as additive rate
points is not settled by any source we could check. It matters a lot: additive would give a
22-year-old +0.042 wOBA where we give +0.013. We use the multiplicative reading. The kink at 29
is a two-slope approximation of an aging curve either way, in both ours and Marcel's.

**5. Age is derived, not looked up.** `age_target` is Savant's season age for the lag-1 season
plus one. Tango uses age as of July 1 of the projected season. These agree within a year, and
disagree by one for players born near the cutoff.

**6. The league rate is close to Marcel's, by construction.** The 5/4/3 recency weighting of the
three lag seasons' league wOBA matches Tango's weighting of the league line. The difference is in
*which* league -- see the next item.

**7. The league average is our snapshot's average, and it has a composition break at 2022.**
`lg_woba` is the PA-weighted mean wOBA of every row in the Savant batter leaderboard for that
season, pulled with `type=batter, min=1`. That leaderboard includes pitchers' batting lines
through 2021 and effectively stops including them with the universal DH in 2022. The size of the
break, measured as the gap between the all-rows league wOBA and the same figure over players
with 100+ PA:

| Seasons | all rows | 100+ PA rows | gap |
|---|---|---|---|
| 2015-2021 | 0.3128-0.3205 | 0.3204-0.3294 | -0.0076 to -0.0096 |
| 2022-2026 | 0.3097-0.3184 | 0.3127-0.3211 | -0.0027 to -0.0032 |

So our league series sits roughly 5 wOBA points lower before 2022 than it would on a
post-DH-consistent definition. **This is bigger than Marcel.** The same `lg_woba` column
centres the league-relative target the selected forecaster is fitted on and supplies
`lg_woba_hat`, so training rows with targets 2016-2019 and 2021 are centred about 5 points
differently from the 2024-2026 test rows. It is recorded here, in `PROJECT_PLAN.md` and in the
README limitations, and it is **not** fixed: the league definition is part of what the tag pins,
and changing it now would silently rebuild every number in the repo. It is the strongest
candidate for a v2 re-freeze.

**8. A missing lag season gets zero weight, not a league-average fill.** A hitter with one prior
season has a weighted PA of 3000 rather than 7200, so he is regressed 28.6% toward the league
instead of 14.3%. That is the same mechanism a smaller denominator gives in Tango's version, but
implementations that fill the missing seasons with league-average performance at a notional PA
land somewhere different.

**9. The population is the easy half, and this favours our Marcel.** We only forecast players
with at least 100 PA in the prior season. Tango's Marcel projects anyone with a major-league
record and hands players without one the league average. Every case where Marcel is weakest --
rookies, players back from a lost year, 30-PA cups of coffee -- is outside our population
entirely. Criterion 1 is therefore "beats Marcel on established hitters", not "beats Marcel".

**10. Two code paths, and the criterion uses one.** `marcel.project_marcel` projects from the
long season table and is exercised only by tests. `baselines.MarcelBaseline` projects from the
feature frame and is what every fold, every bootstrap and the holdout use. They differ in how
target age is derived -- last observed season plus elapsed years, versus lag-1 age plus one --
which is equivalent under the 100-PA population rule, since a lag-1 season always exists.

**11. No park, league, platoon or minor-league adjustment.** Mostly a similarity: the published
Marcel is park-blind too and regresses to the overall major-league average (with an exception
for AL pitchers that does not apply here). The park limitation is recorded in the README and
deferred to v2 with the park features.

**12. 2020 is treated as an ordinary season.** Neither implementation special-cases it; the PA
weighting down-weights a 60-game season to roughly 37% of a full one on its own. The
`short_lag{k}` flags that exist for this are features of the ridge and are invisible to Marcel.

**13. A PA-bearing season with a missing wOBA would be read as wOBA 0.** `MarcelBaseline`
substitutes 0.0 for a NaN rate when the season has plate appearances. Verified against the
snapshot: there are no such rows -- `pa > 0 and woba is null` has a count of 0 across all 9,804
player-seasons -- so this path is unreachable on this data. It is a latent bug, not an active
one, and it is left alone because the file is frozen.

## How to read criterion 1 after this

"`ridge_rel` beats Marcel" means: beats a single-rate, PA-weighted, 5/4/3, 1200-regression,
two-slope-aged projection, on hitters who took 100 PA in each of the bracketing seasons, with
the league level defined as in difference 7. That is a real baseline and a conventional one, and
it is the one that was pre-registered. It is not the published Marcel, it is probably a little
weaker than the published Marcel, and the write-up says so rather than claiming a win over
Tango's system.

## Sources

The published details were checked against these; note that the ones carrying the specific
constants are third-party or self-described as unverified, which is why differences 3 and 4
record an ambiguity rather than a number.

- [Tango on Baseball archives, Marcel post](https://www.tangotiger.net/archives/stud0346.shtml) --
  the 5/4/3 weighting of the league line scaled to 1200 PA, and league average for players with
  no major-league record.
- [Marcel 2012, tangotiger.net](https://tangotiger.net/marcel/) -- the published projections.
- [Forecasting 2006, The Hardball Times](https://tht.fangraphs.com/forecasting-2006/) -- Tango's
  description of regressing by playing time.
- [But I Regress..., The Hardball Times](https://tht.fangraphs.com/but-i-regress/) -- regression
  toward the overall major-league average.
- [Evaluating the 2014 Projection Systems](https://tht.fangraphs.com/evaluating-the-2014-projection-systems/)
  -- Marcel does not project players without major-league experience.
- [marcelR](https://github.com/bdilday/marcelR) -- a third-party reimplementation quoting 5/4/3
  with 100 PA of regression for batting, which is the normalised-scale reading of difference 3.
- [Monkeying around with Fantasy Baseball Projections](https://fansided.com/2013/01/12/marcels-fantasy-baseball-projections/)
  -- the `r = PA / (PA + 1200)` form, with the author noting they had not verified it.
