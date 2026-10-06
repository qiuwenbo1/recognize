# SPDX-FileCopyrightText: 2026 Recognize contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Offline proposals only; no database, network, or automatic assignments."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from adaptive import (
    FEATURES,
    VERSION,
    Gallery,
    assign_content_groups,
    build_clusters,
    eligible,
    main,
    valid,
)


def load_snapshot(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != "recognize-adaptive-snapshot-v1":
        raise ValueError("Unsupported snapshot schema")
    if (
        not isinstance(data.get("embedding_model"), str)
        or not data["embedding_model"].strip()
    ):
        raise ValueError("A specific embedding model identifier is required")
    rows = data["faces"]
    if not rows or len({r["user_id"] for r in rows}) != 1:
        raise ValueError("A nonempty snapshot of exactly one user is required")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate detection IDs")
    for r in rows:
        r.setdefault("content_key", "file:" + str(r["file_id"]))
        if not r["content_key"]:
            r["content_key"] = "file:" + str(r["file_id"])
    data["faces"] = sorted(rows, key=lambda r: r["id"])
    return data


def load_model(path, embedding_model):
    model = json.loads(Path(path).read_text(encoding="utf-8"))
    if model.get("version") != VERSION or model.get("features") != FEATURES:
        raise ValueError("Unsupported scoring model or feature order")
    if model.get("embedding_model") != embedding_model:
        raise ValueError("Scoring model was trained for a different embedding model")
    for field, shape in [
        ("mean", (5,)),
        ("scale", (5,)),
        ("coef", (1, 5)),
        ("intercept", (1,)),
    ]:
        a = np.asarray(model[field], dtype=float)
        if a.shape != shape or not np.isfinite(a).all():
            raise ValueError("Invalid scoring model field: " + field)
    if min(model["scale"]) <= 0 or not 0 <= model["cutoff"] <= 1.01:
        raise ValueError("Invalid scale or cutoff")
    return model


def predict(result, model):
    features = np.asarray(result["features"], dtype=float)
    if not np.isfinite(features).all():
        raise ValueError("Nonfinite scoring features")
    z = float(
        ((features - model["mean"]) / model["scale"]) @ np.asarray(model["coef"])[0]
        + model["intercept"][0]
    )
    return float(1 / (1 + np.exp(-np.clip(z, -700, 700))))


def propose(data, model, output):
    rows = [r for r in data["faces"] if valid(r)]
    assign_content_groups(rows)
    gallery = Gallery(
        [
            r
            for r in rows
            if (r["cluster_id"] or 0) > 0 and r.get("reference_eligible", True)
        ]
    )
    decisions = []
    for q in rows:
        if not eligible(q):
            continue
        result = gallery.query(q)
        if result is None:
            continue
        score = predict(result, model)
        result.update(id=q["id"], score=score, write_authorized=False)
        result["status"] = (
            "suggested"
            if score >= model["cutoff"]
            and result["best"]["name"].strip()
            and result["gap"] > 1e-6
            else "review"
        )
        decisions.append(result)
    scores = {r["id"]: r for r in decisions}
    # Name proposals take precedence so the two review lists are disjoint.
    accepted = {r["id"] for r in decisions if r["status"] == "suggested"}
    clusters = build_clusters(
        [r for r in rows if eligible(r) and r["id"] not in accepted],
        scores,
        model["cutoff"],
    )
    (output / "suggestions.jsonl").write_text(
        "".join(
            json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in decisions
        ),
        encoding="utf-8",
    )
    (output / "clusters.json").write_text(
        json.dumps(clusters, ensure_ascii=False, allow_nan=False, indent=2),
        encoding="utf-8",
    )
    (output / "conflicts.json").write_text(
        json.dumps(gallery.conflicts), encoding="utf-8"
    )
    summary = dict(
        mode="proposals-only",
        database_writes=0,
        evaluated_unassigned=len(decisions),
        named_suggestions=len(accepted),
        candidate_clusters=len(clusters),
        excluded_vectors=len(data["faces"]) - len(rows),
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["train", "propose"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True, help="New private output directory")
    parser.add_argument("--model", help="Frozen model.json, required for propose")
    args = parser.parse_args()
    if args.action == "propose" and not args.model:
        parser.error("propose requires --model")
    data = load_snapshot(args.input)
    os.umask(0o077)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    if args.action == "train":
        main(data, output)
        model_path = output / "model.json"
        model = json.loads(model_path.read_text(encoding="utf-8"))
        model["embedding_model"] = data["embedding_model"]
        model_path.write_text(json.dumps(model, indent=2), encoding="utf-8")
    else:
        propose(data, load_model(args.model, data["embedding_model"]), output)
    manifest = {
        "version": VERSION,
        "embedding_model": data["embedding_model"],
        "input_sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        "mode": "proposals-only",
        "database_writes": 0,
    }
    if args.model:
        manifest["model_sha256"] = hashlib.sha256(
            Path(args.model).read_bytes()
        ).hexdigest()
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    run()
