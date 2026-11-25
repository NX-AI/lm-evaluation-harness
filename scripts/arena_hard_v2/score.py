"""Arena-Hard v2.0 scores from pairwise judgments (official scoring of lmarena/arena-hard-auto).

    python scripts/arena_hard_v2/score.py --judgments JUDGE_DIR [JUDGE_DIR ...] \
        --answers ANSWER_DIR [ANSWER_DIR ...] --category hard_prompt [--style-control] [--out scores.json]

JUDGE_DIR holds one ``<model>.jsonl`` of judgments per model, all from the same judge;
ANSWER_DIR holds ``<model>.jsonl`` answer files with style ``metadata`` (needed for
``--style-control`` only, and must include the category's baseline).

Scores, as in the official ``show_result.py``:
* plain: mean battle score against the category's baseline over 100 bootstrap resamples,
  90% interval from the 5th/95th percentiles; a ``>>`` verdict counts as 3 battles;
* ``--style-control``: Bradley-Terry fit over all models with the answer-length and markdown
  features (``-f markdown length``), median and 5th/95th percentile of the bootstrap win
  probabilities against the baseline. The fit is joint, so the scores depend on which models'
  judgments are loaded.

Adapted from arena-hard-auto (https://github.com/lmarena/arena-hard-auto, commit 48e797c1,
``show_result.py`` and ``utils/math_utils.py``), Copyright the arena-hard-auto authors, licensed
under the Apache License 2.0. Changes: judgments and answers are read from the given directories
instead of ``data/<benchmark>``; one category per call (upstream's category loop filters
cumulatively); bootstrap seeds are fixed; results are returned as a dict and written as JSON;
the label mapping, battle weighting, bootstrap, feature construction and Bradley-Terry fit are
unchanged.
"""

import argparse
import json
import os
from glob import glob

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BASELINES = {
    "hard_prompt": "o3-mini-2025-01-31",
    "coding": "o3-mini-2025-01-31",
    "math": "o3-mini-2025-01-31",
    "creative_writing": "gemini-2.0-flash-001",
}


# ----------------------------------------------------------------------------- show_result.py
def load_judgments(judgment_dirs, weight=3):
    dfs = [pd.read_json(f, lines=True) for d in judgment_dirs for f in sorted(glob(os.path.join(d, "*.jsonl")))]
    data = pd.concat(dfs).reset_index(drop=True)
    null_indices = data.games.map(
        lambda x: x[0] is None or x[1] is None or x[0]["score"] is None or x[1]["score"] is None
    )
    _data = data[~null_indices].reset_index(drop=True)
    label_to_score = {
        "A>B": [1],
        "A>>B": [1] * weight,
        "A=B": [0.5],
        "A<<B": [0] * weight,
        "A<B": [0],
        "B>A": [0],
        "B>>A": [0] * weight,
        "B=A": [0.5],
        "B<<A": [1] * weight,
        "B<A": [1],
    }
    # game 0: baseline is A, model is B; game 1: model is A, baseline is B
    _data["scores"] = _data.games.map(
        lambda x: label_to_score[x[1]["score"]] + [1 - s for s in label_to_score[x[0]["score"]]]
    )
    battles = _data[["uid", "model", "category", "scores"]].explode("scores").reset_index(drop=True)
    return battles, int(null_indices.sum())


def get_model_style_metadata(answer_dirs):
    model_metadata = {}
    for d in answer_dirs:
        for file in sorted(glob(os.path.join(d, "*.jsonl"))):
            df = pd.read_json(file, lines=True)
            model_metadata[df.iloc[0]["model"]] = df.set_index("uid")["metadata"].to_dict()
    return model_metadata


def leaderboard(mean_scores, lower_scores, upper_scores, baseline=None):
    table = mean_scores.merge(lower_scores, on="model").merge(upper_scores, on="model")
    rows = {r.model: dict(score=100 * r.scores, lower=100 * r.lower, upper=100 * r.upper) for r in table.itertuples()}
    if baseline:
        rows[baseline] = dict(score=50.0, lower=50.0, upper=50.0)
    return dict(sorted(rows.items(), key=lambda kv: -kv[1]["score"]))


def plain_scores(battles, category, seed=0):
    baseline = BASELINES[category]
    _battles = battles.drop(columns=["category"])[["model", "scores"]]
    _battles["model"] = _battles["model"].map(lambda x: x.split("/")[-1])
    rng = np.random.RandomState(seed)
    bootstraps = pd.concat(
        [
            _battles.groupby("model").sample(frac=1.0, replace=True, random_state=rng).groupby("model").mean()
            for _ in range(100)
        ]
    )
    bootstraps["scores"] = bootstraps["scores"].astype(float)
    mean_scores = bootstraps.groupby("model").mean().reset_index()
    lower_scores = bootstraps.groupby("model").quantile(0.05).reset_index().rename(columns={"scores": "lower"})
    upper_scores = bootstraps.groupby("model").quantile(0.95).reset_index().rename(columns={"scores": "upper"})
    raw = _battles.assign(scores=_battles.scores.astype(float)).groupby("model").scores.mean()
    board = leaderboard(mean_scores, lower_scores, upper_scores, baseline)
    for model, value in raw.items():
        board[model]["raw_mean"] = 100 * value
    return board


def style_controlled_scores(battles, answer_dirs, category, control_features=("markdown", "length"), seed=0):
    baseline = BASELINES[category]
    style_metadata = get_model_style_metadata(answer_dirs)
    model_features = battles.apply(lambda row: style_metadata[row["model"]][row["uid"]], axis=1).tolist()
    baseline_features = battles.apply(lambda row: style_metadata[BASELINES[row["category"]]][row["uid"]], axis=1).tolist()
    battles = battles.copy()
    battles["model"] = battles["model"].map(lambda x: x.split("/")[-1])
    model_feature_tensor = torch.tensor(
        [[v if isinstance(v, int) else sum(v.values()) for k, v in metadata.items()] for metadata in model_features],
        dtype=torch.float32,
    )
    baseline_feature_tensor = torch.tensor(
        [[v if isinstance(v, int) else sum(v.values()) for k, v in metadata.items()] for metadata in baseline_features],
        dtype=torch.float32,
    )
    final_feature_tensor = torch.zeros_like(model_feature_tensor)
    final_feature_tensor[:, 0] = (model_feature_tensor[:, 0] - baseline_feature_tensor[:, 0]) / (
        model_feature_tensor[:, 0] + baseline_feature_tensor[:, 0]
    )
    model_md_density = model_feature_tensor[:, 1:] / (model_feature_tensor[:, :1] + 1)
    baseline_md_density = baseline_feature_tensor[:, 1:] / (baseline_feature_tensor[:, :1] + 1)
    assert not model_md_density.isnan().any()
    assert not baseline_md_density.isnan().any()
    final_feature_tensor[:, 1:] = (model_md_density - baseline_md_density) / (
        model_md_density + baseline_md_density + 1
    )
    assert not final_feature_tensor.isnan().any()
    normalized_feature_tensor = (final_feature_tensor - torch.mean(final_feature_tensor, axis=0)) / torch.std(
        final_feature_tensor, axis=0
    )
    assert not normalized_feature_tensor.isnan().any()
    outcomes = torch.tensor(battles.scores.tolist())
    assert not outcomes.isnan().any()
    one_hot, unique_models = one_hot_encode(battles.model.tolist(), baseline=baseline)
    all_features = torch.cat([one_hot, normalized_feature_tensor], dim=1)
    if "length" in control_features and "markdown" in control_features:
        num_features = 4
    elif "length" in control_features:
        all_features = all_features[:, :1]
        num_features = 1
    elif "markdown" in control_features:
        all_features = all_features[:, 1:]
        num_features = 3
    else:
        raise ValueError("control_features must include 'length' and/or 'markdown'")
    coefs, _ = bootstrap_pairwise_model(all_features, outcomes, loss_type="bt", seed=seed)
    _coefs = coefs[:, :-num_features]
    table = pd.DataFrame(
        columns=unique_models,
        data=to_winrate_probabilities(_coefs, unique_models, baseline_model=baseline).tolist(),
    )
    board = leaderboard(
        table.quantile(0.5).to_frame("scores").reset_index().rename(columns={"index": "model"}),
        table.quantile(0.05).to_frame("lower").reset_index().rename(columns={"index": "model"}),
        table.quantile(0.95).to_frame("upper").reset_index().rename(columns={"index": "model"}),
    )
    feature_coefs = torch.quantile(coefs[:, -num_features:], 0.5, axis=0).tolist()
    return board, feature_coefs


# ----------------------------------------------------------------------------- utils/math_utils.py
class BTModel(nn.Module):
    def __init__(self, num_components):
        super().__init__()
        self.logits = nn.Parameter(nn.init.constant_(torch.empty(num_components), 0.5))

    def forward(self):
        return self.logits, None


def bt_loss(logits, outcomes, **kwargs):
    return F.binary_cross_entropy_with_logits(logits, outcomes.float(), reduction="sum")


def fit_pairwise_model(features, outcomes, loss_type="bt", indices=None, lr=0.1, tol=1e-9, max_epochs=50):
    assert loss_type == "bt"
    if indices is not None:
        features = features[indices]
        outcomes = outcomes[indices]
    assert not features.isnan().any()
    model = BTModel(num_components=features.shape[1])
    optimizer = optim.LBFGS(model.parameters(), lr=lr, max_iter=max_epochs, tolerance_grad=tol, tolerance_change=tol)

    def closure():
        optimizer.zero_grad()
        logits, eta = model()
        loss = bt_loss(logits=features @ logits, outcomes=outcomes, eta=eta)
        loss.backward()
        return loss

    optimizer.step(closure)
    logits, eta = model()
    return logits.detach(), None


def bootstrap_pairwise_model(features, outcomes, loss_type="bt", num_round=100, seed=0):
    rng = np.random.RandomState(seed)
    boot_idxs = rng.randint(low=0, high=features.shape[0], size=(num_round, features.shape[0]))
    results = [fit_pairwise_model(features, outcomes, loss_type, boot_idxs[i]) for i in range(num_round)]
    return torch.stack([r[0] for r in results]), None


def one_hot_encode(items, baseline="o3-mini-2025-01-31"):
    unique_items = sorted(set(items + [baseline]))
    item_to_index = {item: idx for idx, item in enumerate(unique_items)}
    one_hot_matrix = []
    for item in items:
        one_hot_vector = [0] * len(unique_items)
        one_hot_vector[item_to_index[item]] = 1
        one_hot_vector[item_to_index[baseline]] = -1
        one_hot_matrix.append(one_hot_vector)
    return torch.tensor(one_hot_matrix, dtype=torch.float32), unique_items


def to_winrate_probabilities(coefs, models, baseline_model="o3-mini-2025-01-31"):
    baseline_idx = models.index(baseline_model)
    exp_coefs = torch.exp(coefs)
    probs = torch.zeros(coefs.shape[0], coefs.shape[1])
    for idx in range(len(models)):
        probs[:, idx] = exp_coefs[:, idx] / (exp_coefs[:, idx] + exp_coefs[:, baseline_idx])
    return probs


# ----------------------------------------------------------------------------- CLI
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--judgments", nargs="+", required=True, help="judgment directories (one judge)")
    parser.add_argument("--answers", nargs="*", default=[], help="answer directories with style metadata")
    parser.add_argument("--category", default="hard_prompt", choices=sorted(BASELINES))
    parser.add_argument("--style-control", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", help="write the scores as JSON here")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    battles, nulls = load_judgments(args.judgments)
    if args.category not in set(battles.category):
        raise SystemExit(f"no judgments for category {args.category}")
    battles = battles[battles.category == args.category].reset_index(drop=True)
    result = dict(category=args.category, baseline=BASELINES[args.category], null_judgments=nulls,
                  style_control=args.style_control, judgments=args.judgments)
    if args.style_control:
        if not args.answers:
            raise SystemExit("--style-control needs --answers with style metadata")
        result["scores"], result["feature_coefs"] = style_controlled_scores(battles, args.answers, args.category,
                                                                           seed=args.seed)
    else:
        result["scores"] = plain_scores(battles, args.category, seed=args.seed)
    print(f"##### {args.category} ({'style control' if args.style_control else 'plain'}), "
          f"{nulls} unparsed judgments dropped #####")
    for model, row in result["scores"].items():
        print(f"{model:45s} {row['score']:5.1f}  (-{row['score'] - row['lower']:.1f} / +{row['upper'] - row['score']:.1f})")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
