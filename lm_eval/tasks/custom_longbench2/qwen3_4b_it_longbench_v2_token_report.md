# Qwen3-4B-Instruct context token lengths on LongBench v2

- Model config: `/nfs-gpu/xlstm-distillation/work_niklas/xlstm-distillation-internal/configs/model/qwen3_4b/qwen3_4b_it_baseline_fft.yaml`
- Tokenizer: `Qwen/Qwen3-4B-Instruct-2507`
- Bucketing is prompt-agnostic and uses only the raw `context` field so different prompt strategies evaluate the exact same documents in each bucket.
- `qwen_4b_it_tokens` and `qwen_4b_it_context_tokens` both store the raw `context` token count for compatibility with the existing task loader.
- `qwen_4b_it_length_bucket` uses context-token buckets `0_16k`, `16k_32k`, `32k_64k`, `64k_128k`, `128k_512k`, `512k_1024k`.

## Overall (context tokens)

```json
{
  "max": 4163702,
  "mean": 260391.09343936382,
  "min": 9975,
  "n": 503,
  "p50": 99412,
  "p90": 519853,
  "p95": 1148012,
  "p99": 3326997
}
```

## By sub-domain (context tokens)

### Academic

```json
{
  "max": 651543,
  "mean": 90491.53191489361,
  "min": 15179,
  "n": 94,
  "p50": 41684,
  "p90": 212606,
  "p95": 386604,
  "p99": 598738
}
```

### Agent history QA

```json
{
  "max": 64605,
  "mean": 33724.85,
  "min": 24270,
  "n": 20,
  "p50": 30385,
  "p90": 42012,
  "p95": 44239,
  "p99": 60531
}
```

### Code repo QA

```json
{
  "max": 4163702,
  "mean": 1071100.42,
  "min": 24386,
  "n": 50,
  "p50": 503440,
  "p90": 3337101,
  "p95": 3500174,
  "p99": 3899371
}
```

### Detective

```json
{
  "max": 212709,
  "mean": 112816.59090909091,
  "min": 18823,
  "n": 22,
  "p50": 97813,
  "p90": 192378,
  "p95": 193841,
  "p99": 208749
}
```

### Dialogue history QA

```json
{
  "max": 125333,
  "mean": 119593.8947368421,
  "min": 117058,
  "n": 19,
  "p50": 119699,
  "p90": 120881,
  "p95": 121563,
  "p99": 124579
}
```

### Event ordering

```json
{
  "max": 375759,
  "mean": 176880.4,
  "min": 36636,
  "n": 20,
  "p50": 164412,
  "p90": 232717,
  "p95": 348975,
  "p99": 370402
}
```

### Financial

```json
{
  "max": 1740677,
  "mean": 180481.8108108108,
  "min": 12948,
  "n": 37,
  "p50": 114048,
  "p90": 256963,
  "p95": 438043,
  "p99": 1284280
}
```

### Governmental

```json
{
  "max": 960383,
  "mean": 168729.0975609756,
  "min": 10836,
  "n": 41,
  "p50": 134185,
  "p90": 357778,
  "p95": 521067,
  "p99": 799581
}
```

### Knowledge graph reasoning

```json
{
  "max": 3225507,
  "mean": 412388.2,
  "min": 143698,
  "n": 15,
  "p50": 178849,
  "p90": 444394,
  "p95": 1328152,
  "p99": 2846036
}
```

### Legal

```json
{
  "max": 450630,
  "mean": 64818.333333333336,
  "min": 10113,
  "n": 33,
  "p50": 27463,
  "p90": 182026,
  "p95": 225831,
  "p99": 382580
}
```

### Literary

```json
{
  "max": 865154,
  "mean": 186287.83333333334,
  "min": 12527,
  "n": 30,
  "p50": 99969,
  "p90": 515239,
  "p95": 624863,
  "p99": 807247
}
```

### Many-shot learning

```json
{
  "max": 156920,
  "mean": 115840.76190476191,
  "min": 91262,
  "n": 21,
  "p50": 97496,
  "p90": 156920,
  "p95": 156920,
  "p99": 156920
}
```

### Multi-news

```json
{
  "max": 235010,
  "mean": 52093.608695652176,
  "min": 9975,
  "n": 23,
  "p50": 24219,
  "p90": 107137,
  "p95": 174761,
  "p99": 223256
}
```

### New language translation

```json
{
  "max": 1148012,
  "mean": 580749.0,
  "min": 255047,
  "n": 20,
  "p50": 301686,
  "p90": 1148012,
  "p95": 1148012,
  "p99": 1148012
}
```

### Table QA

```json
{
  "max": 2302195,
  "mean": 543243.0555555555,
  "min": 18887,
  "n": 18,
  "p50": 327174,
  "p90": 1420941,
  "p95": 2061652,
  "p99": 2254086
}
```

### User guide QA

```json
{
  "max": 1474151,
  "mean": 185394.85,
  "min": 12869,
  "n": 40,
  "p50": 111670,
  "p90": 319221,
  "p95": 702487,
  "p99": 1331369
}
```

## Non-empty context-token buckets by sub-domain

### Academic

```json
{
  "0_16k": 7,
  "128k_512k": 12,
  "16k_32k": 33,
  "32k_64k": 25,
  "512k_1024k": 3,
  "64k_128k": 14
}
```

### Agent history QA

```json
{
  "16k_32k": 12,
  "32k_64k": 8
}
```

### Code repo QA

```json
{
  "128k_512k": 10,
  "16k_32k": 3,
  "32k_64k": 3,
  "512k_1024k": 25,
  "64k_128k": 9
}
```

### Detective

```json
{
  "128k_512k": 6,
  "16k_32k": 1,
  "64k_128k": 15
}
```

### Dialogue history QA

```json
{
  "64k_128k": 19
}
```

### Event ordering

```json
{
  "128k_512k": 16,
  "32k_64k": 1,
  "64k_128k": 3
}
```

### Financial

```json
{
  "0_16k": 2,
  "128k_512k": 15,
  "16k_32k": 3,
  "32k_64k": 3,
  "512k_1024k": 1,
  "64k_128k": 13
}
```

### Governmental

```json
{
  "0_16k": 2,
  "128k_512k": 19,
  "16k_32k": 9,
  "32k_64k": 5,
  "512k_1024k": 2,
  "64k_128k": 4
}
```

### Knowledge graph reasoning

```json
{
  "128k_512k": 14,
  "512k_1024k": 1
}
```

### Legal

```json
{
  "0_16k": 8,
  "128k_512k": 4,
  "16k_32k": 10,
  "32k_64k": 8,
  "64k_128k": 3
}
```

### Literary

```json
{
  "0_16k": 2,
  "128k_512k": 11,
  "16k_32k": 3,
  "32k_64k": 4,
  "512k_1024k": 3,
  "64k_128k": 7
}
```

### Many-shot learning

```json
{
  "128k_512k": 5,
  "64k_128k": 16
}
```

### Multi-news

```json
{
  "0_16k": 6,
  "128k_512k": 2,
  "16k_32k": 7,
  "32k_64k": 4,
  "64k_128k": 4
}
```

### New language translation

```json
{
  "128k_512k": 13,
  "512k_1024k": 7
}
```

### Table QA

```json
{
  "128k_512k": 9,
  "16k_32k": 1,
  "32k_64k": 1,
  "512k_1024k": 5,
  "64k_128k": 2
}
```

### User guide QA

```json
{
  "0_16k": 2,
  "128k_512k": 13,
  "16k_32k": 5,
  "32k_64k": 6,
  "512k_1024k": 3,
  "64k_128k": 11
}
```

## Raw token counts

- JSONL: `qwen3_4b_it_longbench_v2_tokens.jsonl`
