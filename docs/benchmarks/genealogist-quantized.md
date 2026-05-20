# Genealogist Quantized Model Benchmark — ARG-91

Compare `qwen3:32b` (full precision, ~20 GB VRAM) against `qwen3:32b-q4_K_M`
(~11 GB VRAM) on the genealogy reasoning task to determine whether the quantized
model is production-suitable and to verify that a larger `num_ctx` is stable.

---

## Prerequisites

```bash
ollama pull qwen3:32b          # full-precision baseline
ollama pull qwen3:32b-q4_K_M  # quantized candidate
```

Both models must be available locally before running.

---

## Running the Benchmark

Run the script **twice** — once per model — and compare the two output files:

```bash
# Baseline (full precision)
uv run python scripts/benchmark_genealogist_quantized.py \
    --model qwen3:32b \
    --num-ctx 3072 \
    --items 15 \
    --out reports/genealogist-full.json

# Quantized candidate
uv run python scripts/benchmark_genealogist_quantized.py \
    --model qwen3:32b-q4_K_M \
    --num-ctx 6144 \
    --items 15 \
    --out reports/genealogist-q4km.json
```

### CLI flags

| Flag | Default | Description |
|---|---|---|
| `--model` | `qwen3:32b` | Ollama model tag |
| `--num-ctx` | `3072` | KV-cache context size |
| `--items` | `15` | How many tech_items to evaluate |
| `--out` | `reports/genealogist-benchmark.json` | Output JSON path |

---

## Item Selection Criteria

The script fetches the `--items` most-recently-inserted `tech_items` from the dev
DB (ordered by `created_at DESC`).  For a meaningful comparison:

- Ensure the DB contains at least 30–50 items spanning both `Mainstream` and `Alpha` categories.
- Use the **same item set** for both model runs (same DB state, same `--items` count).
- A diverse set is preferred — run the crawler a few times beforehand if the DB is sparse.

---

## Prompt

The exact same prompt as `argos.brain.nodes.genealogist._GENEALOGIST_PROMPT` is
used — it is imported directly (not copied) so any future prompt edits are
automatically reflected in benchmark runs.

In benchmark mode the script passes `"(benchmark mode — no similarity context)"`
as the `existing_techs` argument because the benchmark is evaluating raw model
reasoning quality, not the full pipeline.

---

## Scoring Rubric

For each item, score manually after inspecting the two JSON reports:

| Dimension | 1 | 3 | 5 |
|---|---|---|---|
| **Relation correctness** | Wrong or hallucinated relation | Plausible but imprecise | Correct Replace/Enhance/Fork classification |
| **Reasoning coherence** | Incoherent / generic | Partially specific | Concise, specific, grounded in item content |

Aggregate mean scores across all items per model.  A score within 0.5 points on
both dimensions is considered acceptable degradation for the quantized model.

---

## VRAM / num_ctx Probing Procedure

### Reading Metal memory usage

```bash
# While the benchmark is running (new terminal):
sudo powermetrics --samplers gpu_power -i 1000 -n 5

# Or use Activity Monitor → Window → GPU History
# Or check ollama ps output:
ollama ps
```

`ollama ps` shows `VRAM` column in GB and confirms which layers are on CPU
(non-zero `CPU%` means offloading is occurring).

### Stepping up num_ctx

Start at the current baseline (`3072`) and increase in steps of 1024:

```bash
for ctx in 3072 4096 5120 6144 7168 8192; do
    uv run python scripts/benchmark_genealogist_quantized.py \
        --model qwen3:32b-q4_K_M \
        --num-ctx $ctx \
        --items 3 \
        --out reports/ctx-probe-${ctx}.json
    # Check ollama ps VRAM and CPU% immediately after each run
    ollama ps
done
```

The maximum stable `num_ctx` is the largest value where `ollama ps` shows no CPU
layers and total VRAM stays below ~28 GB (leaving headroom for macOS and the
embedder).

### Expected result (M1 Max 32GB)

Based on VRAM calculations:
- `qwen3:32b` (~20 GB): max stable `num_ctx` ≈ 3072–4096 before CPU offload
- `qwen3:32b-q4_K_M` (~11 GB): max stable `num_ctx` ≈ 6144–8192

---

## Report Format

Each run produces a JSON file:

```json
{
  "model": "qwen3:32b-q4_K_M",
  "num_ctx": 6144,
  "items_evaluated": 15,
  "results": [
    {
      "item_id": "<uuid>",
      "title": "Some Tech",
      "relation_type": "Replace",
      "reason": "brief explanation",
      "elapsed_s": 4.321
    },
    {
      "item_id": "<uuid>",
      "title": "Another Tech",
      "relation_type": null,
      "reason": "",
      "elapsed_s": 2.1,
      "error": "GPU OOM or parse error message"
    }
  ]
}
```

`elapsed_s` measures wall time from the HTTP request start to the parsed
response (includes first-token latency + generation).  An error record has an
`"error"` key with the exception message; the run continues regardless.

---

## Interpreting Results

1. Compare `relation_type` distributions — both models should output a similar
   mix of `Replace` / `Enhance` / `Fork` / `null`.
2. Compare mean `elapsed_s` — the quantized model should be faster.
3. Manually spot-check `reason` text for coherence using the scoring rubric.
4. If quantized model scores within 0.5 points on both dimensions **and** `num_ctx`
   can be bumped to ≥ 6144 without CPU offload → update `genealogist.model` in
   `~/.config/argos/config.toml` to `qwen3:32b-q4_K_M` and set `genealogist.num_ctx`
   to the confirmed stable value.
