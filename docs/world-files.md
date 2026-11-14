# World files

Describe your own inventory in a file. The complete field reference is at the top of [`world.py`](../world.py).

## Your own inventory: world files

The products, storage and tasks of the pharma world are not hard-coded: they live in a **world file**, [`worlds/pharma.yaml`](../worlds/pharma.yaml). Describe your own inventory in the same format and every agent, the news and `eval.py` work on it:

```bash
uv run python eval.py --world examples/worlds/grocery.yaml --agent baseline --episodes 20
uv run python eval.py --world examples/worlds/grocery.yaml --agent oracle   --episodes 20 --news 1
```

A world file has four parts (the complete reference is at the top of [`world.py`](../world.py), and [`examples/worlds/grocery.yaml`](../examples/worlds/grocery.yaml) is a worked example):

```yaml
name: my_store
business: a neighbourhood grocery store   # who the agent works for (used in the prompts)
days: 45
storage:                                  # any number of pools, any names
  fridge:  {capacity_days: 10, cold_chain: true}
  freezer: {capacity_days: 14, cold_chain: true}
  shelf:   {capacity_days: 21}
products:                                 # any number, any names
  milk: {storage: fridge, base_demand: [80, 120], base_lead_time: [1, 2],
         starting_inventory: [100, 200], stockout_penalty: 50, waste_penalty: 8,
         name: fresh milk, description: "sold daily, spoils fast"}
news:                                     # the reasons news gives for disruptions
  delay_causes: [a flooded motorway, a strike at the distribution centre]
  surge_causes: {milk: the holiday season}
tasks:                                    # the same products under different events
  holiday_rush:
    events:
      - {type: demand_step,    product: milk, start_day: 20, end_day: 27, multiplier: [1.8, 2.2]}
      - {type: demand_wave,    product: milk, peak_day: 24, amplitude: [0.5, 1.0], width: 4}
      - {type: supplier_delay, product: milk, start_day: 15, end_day: 24, multiplier: [2, 3]}
```

- **Amounts** are a number or a `[low, high]` range drawn once per episode from the seed, so a seed always gives the same episode and thousands of seeds give thousands of variations.
- **Events:** `supplier_delay` (lead time × multiplier for orders placed in a window), `demand_wave` (a bell-shaped rise around a peak day) and `demand_step` (demand × multiplier in a window: a promotion, or a closure below 1). News announces them like the built-in tasks' events.
- **Mistakes are reported with where they are**: an unknown product, a range with low > high, a day past the end of the episode, a misspelt key or a key written twice.
- **Storage pools** each hold `capacity_days` of their products' base demand, and a delivery that does not fit in its own pool is wasted even when another pool has room. With one cold-chain pool and one other pool (like pharma) the LLM report keeps the original `cold_*`/`ambient_*` keys, so existing prompts and trained models are unaffected; any other layout is reported pool by pool, with each SKU's `storage_pool`.
- **Where and when can be drawn per seed.** With fixed products and days every episode of a task is disrupted the same way, and a fixed timetable can replace reading the news. A task can instead `draw` its products (`struck: any`, or one of a list) and take day ranges (`start_day: [8, 25]`, `duration: [8, 12]`, `peak_day: [18, 34]`), and can have its own storage size (`storage_days: {shelf: 14}`). Products and days come from their own generator, so a task and its drawn twin share demand, lead times and event sizes per seed. In the grocery example's `surprise_disruptions` task (100 test seeds, level-2 news) keeping 3 extra days of everything scores 0.939, *below* the baseline's 0.951, while the news plan scores 0.969 and closes 47% of the gap to the oracle (0.989).
- **Prompts and news are written from the world.** The LLM is told whose inventory it manages (`business`), gets a list of the products with their plain names and descriptions, and reads news that names those products and the causes you list: *"Planning note: a heat wave is on the way. Sales of ice cream could roughly double in about 7 days and stay high for roughly 2 weeks."* A model may answer with a product's id, its name or one of its `aliases`. All of these fields are optional. The built-in pharma world is the exception: it keeps its original wording, byte for byte, so existing results and trained models stay valid; an edited copy of `pharma.yaml` is a world of its own and gets its words from the file.
- The pharma world builds exactly the original tasks (checked config by config, so every published score stands). In code: `world.load_world(path)` and `env.reset(world=..., task_name=..., seed=...)`.

`--random-scenarios` remains the pharma world's own, older form of this (its tasks drawn by code in `tasks.py`); other worlds use `draw` and day ranges in the file.

**Not yet (planned):** the local LLM runner and dashboard picking a world; training (`training/`) on worlds other than pharma.
