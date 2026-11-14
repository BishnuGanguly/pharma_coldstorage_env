# Scoring

How a day and an episode are scored. The same formulas apply to every world.

## Reward and score

### Per-Step Reward

At each step the reward is:

```
step_reward = weighted_fill_today − waste_fraction_today

weighted_fill_today = Σ stockout_penalty_i × demand_fulfilled_today_i / Σ stockout_penalty_i
```

- `demand_fulfilled_today` per SKU is the fraction of true demand served (0.0 = full stockout, 1.0 = fully served).
- The fill rates are averaged with each SKU's `stockout_penalty` as the weight, so a shortage costs in proportion to how critical the medicine is. With the built-in catalogue (insulin 100, BP medication 60, HCQ 35, paracetamol 20, vitamins 5; total 220), a full day without insulin costs 100/220 = 0.45 of the day's reward, and a full day without vitamins 5/220 = 0.02. This matches the prompt's instruction to protect high-`stockout_penalty` SKUs first. (Before, all SKUs counted equally: an insulin stockout cost the same as a vitamins one.)
- `waste_fraction_today` is the share of the day's deliveries rejected because their storage pool was full. It is computed per SKU that received a delivery, then averaged using each SKU's `waste_penalty` as the weight:

  ```
  waste_fraction_today = Σ waste_penalty_i × (wasted_i / delivered_i) / Σ waste_penalty_i
  ```

  It is 0 when nothing arrived, and 1 when every delivery was rejected, so the step reward lies in [-1, 1]. Wasting 1 unit of a 3,000-unit delivery costs almost nothing; wasting half of an insulin delivery costs far more than wasting half of a vitamins delivery.

### Final Episode Score

```
final_score = clip( sum of step rewards / no_of_days , 0, 1 )
            = clip( mean over days of (weighted fill today − waste_fraction_today) , 0, 1 )
```

- The episode score is simply the **average daily step reward**, so an agent is evaluated on exactly what it is rewarded for each day (useful when training with RL).
- An agent that orders nothing serves almost no demand and scores about 0. Waste only ever subtracts, so a day without deliveries earns nothing extra.
- Days not played (an episode cut short) count as 0.
- `tasks.compute_final_score` rebuilds the sum from running totals in the state: `demand_fulfilled_cumulative` per SKU, and `inventory_excess_cumulative`, which adds `1 − waste_fraction_today` each day.
