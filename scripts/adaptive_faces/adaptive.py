# SPDX-FileCopyrightText: 2026 Recognize contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Read-only, deterministic adaptive face proposals. Never writes Nextcloud DB."""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import collections, hashlib, json, math, pathlib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

VERSION = "adaptive-proposals-v1"
FEATURES = [
    "distance",
    "competitor_gap",
    "local_impostor_gap",
    "support",
    "reference_count",
]


def order(x):
    return hashlib.sha256(str(x).encode()).hexdigest()


def valid(r):
    try:
        v = np.asarray(r["vector"], dtype=float)
        return (
            v.shape == (512,)
            and np.isfinite(v).all()
            and abs(np.linalg.norm(v) - 1) < 0.001
        )
    except (TypeError, ValueError, KeyError):
        return False


def eligible(r):
    return (
        (r["cluster_id"] or 0) <= 0
        and (r["threshold"] or 0) <= 0
        and r["width"] >= 0.03
        and r["height"] >= 0.03
        and valid(r)
    )


def distances(a, b):
    return np.sqrt(np.maximum(0, 2 - 2 * np.clip(a @ b.T, -1, 1)))


class Union:
    def __init__(self, n):
        self.p = list(range(n))

    def root(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def join(self, a, b):
        self.p[self.root(a)] = self.root(b)


def assign_content_groups(rows):
    """Leakage groups: exact image content, same file, and near-duplicate faces.
    These are proxies, not a claim of verified shoot/session separation.
    """
    uf = Union(len(rows))
    hashes = {}
    files = {}
    mat = np.array([r["vector"] for r in rows])
    for i, r in enumerate(rows):
        for key, table in [(r["content_key"], hashes), (r["file_id"], files)]:
            if key in table:
                uf.join(i, table[key])
            else:
                table[key] = i
    for lo in range(0, len(rows), 128):
        ds = distances(mat[lo : lo + 128], mat)
        for a, b in zip(*np.where(ds < 0.35)):
            if lo + a < b:
                uf.join(lo + int(a), int(b))
    roots = collections.defaultdict(list)
    for i, r in enumerate(rows):
        roots[uf.root(i)].append(r["id"])
    for i, r in enumerate(rows):
        r["content_group"] = min(roots[uf.root(i)])


class Gallery:
    def __init__(self, rows):
        # Do not permit the same descriptor under conflicting identities to vote.
        byvec = collections.defaultdict(list)
        for r in rows:
            if (r["cluster_id"] or 0) > 0 and valid(r):
                key = np.asarray(r["vector"], dtype=np.float32).tobytes()
                byvec[key].append(r)
        self.conflicts = [
            sorted(x["id"] for x in rs)
            for rs in byvec.values()
            if len({x["cluster_id"] for x in rs}) > 1
        ]
        self.rows = []
        for rs in byvec.values():
            if len({x["cluster_id"] for x in rs}) == 1:
                # Keep duplicate metadata for per-query same-photo exclusions.
                self.rows.extend(rs)
        self.mat = np.array([r["vector"] for r in self.rows], dtype=np.float64)
        self.labels = np.array([r["cluster_id"] for r in self.rows])
        self.local_negative = np.full(len(self.rows), np.nan)
        for lo in range(0, len(self.rows), 128):
            ds = distances(self.mat[lo : lo + 128], self.mat)
            ds[self.labels[lo : lo + 128, None] == self.labels[None, :]] = np.inf
            self.local_negative[lo : lo + 128] = ds.min(axis=1)

    def query(self, q, exclude_group=False):
        if not self.rows:
            return None
        ds = distances(np.array([q["vector"]]), self.mat)[0]
        for i, r in enumerate(self.rows):
            if (
                r["id"] == q["id"]
                or r["file_id"] == q["file_id"]
                or r["content_key"] == q["content_key"]
                or ds[i] < 1e-6
                or (exclude_group and r["content_group"] == q["content_group"])
            ):
                ds[i] = np.inf
        candidates = []
        for c in sorted(set(self.labels)):
            ix = np.where((self.labels == c) & np.isfinite(ds))[0]
            if not len(ix):
                continue
            ix = sorted(ix, key=lambda i: (ds[i], self.rows[i]["id"]))
            seen = set()
            independent = []
            for i in ix:
                group = self.rows[i]["content_group"]
                if group not in seen:
                    independent.append(i)
                    seen.add(group)
            first = ix[0]
            candidates.append(
                {
                    "cluster": int(c),
                    "name": self.rows[first]["title"] or "",
                    "distance": float(ds[first]),
                    "reference": self.rows[first]["id"],
                    "local_negative": float(self.local_negative[first]),
                    "support": int(
                        sum(ds[i] <= ds[first] + 0.10 for i in independent[:5])
                    ),
                    "reference_count": len(independent),
                    "references": [self.rows[i]["id"] for i in independent[:3]],
                }
            )
        candidates.sort(key=lambda s: (s["distance"], s["cluster"]))
        if len(candidates) < 2:
            return None
        a, b = candidates[:2]
        gap = b["distance"] - a["distance"]
        # Unnamed winners compete, but cannot be automatically named.
        features = [
            a["distance"],
            gap,
            a["local_negative"] - a["distance"],
            math.log1p(a["support"]),
            math.log1p(a["reference_count"]),
        ]
        return {"best": a, "runner_up": b, "gap": gap, "features": features}


def make_split(rows, confirmed=None):
    confirmed = confirmed or {}
    named = [
        r for r in rows if (r["cluster_id"] or 0) > 0 and (r["title"] or "").strip()
    ]
    clusters = sorted({r["cluster_id"] for r in named}, key=order)
    # Whole identities withheld: separate unknown identities for each partition.
    unknown = {
        c: ["train", "calibration", "test"][i % 3]
        for i, c in enumerate(clusters)
        if i % 5 == 0
    }
    case_groups = {r["content_group"] for r in rows if r["id"] in confirmed}
    groups = sorted({r["content_group"] for r in rows}, key=order)
    split = {
        g: (
            "gallery"
            if i % 10 < 5
            else "train" if i % 10 < 7 else "calibration" if i % 10 < 9 else "test"
        )
        for i, g in enumerate(groups)
    }
    # A group photo can contain both known and withheld identities.
    overrides = collections.defaultdict(set)
    for r in rows:
        if r["cluster_id"] in unknown:
            overrides[r["content_group"]].add(unknown[r["cluster_id"]])
    for group, partitions in overrides.items():
        if len(partitions) == 1:
            split[group] = next(iter(partitions))
        else:
            case_groups.add(group)
    refs = []
    tests = collections.defaultdict(list)
    for r in rows:
        if r["content_group"] in case_groups or not r.get("reference_eligible", True):
            continue
        c = r["cluster_id"] or 0
        if c <= 0:
            continue
        if c in unknown:
            tests[unknown[c]].append((r, None))
            continue
        if split[r["content_group"]] == "gallery":
            refs.append(r)
        elif (r["title"] or "").strip():
            tests[split[r["content_group"]]].append((r, c))
    labels = {r["cluster_id"] for r in refs}
    for s in tests:
        # One observation per content group/person: don't inflate burst evidence.
        unique = {}
        for r, c in sorted(tests[s], key=lambda x: order(x[0]["id"])):
            if c is None or c in labels:
                unique.setdefault((r["content_group"], c), (r, c))
        tests[s] = list(unique.values())
    return refs, tests, case_groups


def evaluate(records, cutoff):
    counts = collections.Counter(total=len(records))
    known = unknown = 0
    for r in records:
        known += r["expected"] is not None
        unknown += r["expected"] is None
        accepted = (
            r["score"] >= cutoff and bool(r["best"]["name"].strip()) and r["gap"] > 1e-6
        )
        if accepted:
            counts["accepted"] += 1
            counts["correct" if r["best"]["cluster"] == r["expected"] else "wrong"] += 1
            if r["expected"] is None:
                counts["unknown_false_accept"] += 1
        elif r["expected"] is not None:
            counts["known_rejected"] += 1
    return dict(counts, known=known, unknown=unknown)


def build_clusters(rows, scorer, cutoff, edge=0.70):
    """Global mutual-kNN + complete-link guard. No weak-link transitive chaining.
    min size three is independent of paging. Proposals only.
    """
    if len(rows) < 3:
        return []
    mat = np.array([r["vector"] for r in rows])
    n = len(rows)
    neighbors = []
    for lo in range(0, n, 128):
        ds = distances(mat[lo : lo + 128], mat)
        for j, d in enumerate(ds):
            d[lo + j] = np.inf
            ix = np.argsort(d, kind="stable")[:20]
            neighbors.append({int(i): float(d[i]) for i in ix if d[i] <= edge})
    edges = sorted(
        (d, i, j)
        for i, ns in enumerate(neighbors)
        for j, d in ns.items()
        if i < j and i in neighbors[j]
    )
    uf = Union(n)
    members = {i: [i] for i in range(n)}
    for d, i, j in edges:
        a, b = uf.root(i), uf.root(j)
        if a == b:
            continue
        aa, bb = members[a], members[b]
        # Distinct detections on one photo require review (mirrors/collages included).
        if {rows[x]["file_id"] for x in aa} & {rows[x]["file_id"] for x in bb}:
            continue
        if distances(mat[aa], mat[bb]).max() > edge:
            continue
        uf.join(a, b)
        members[b] = aa + bb
        del members[a]
    out = []
    for inds in members.values():
        if len(inds) < 3:
            continue
        ids = sorted(rows[i]["id"] for i in inds)
        votes = collections.defaultdict(list)
        for i in inds:
            s = scorer.get(rows[i]["id"])
            if s and s["best"]["name"].strip():
                votes[s["best"]["cluster"]].append(s)
        suggestion = None
        if votes:
            winner = max(votes, key=lambda c: len(votes[c]))
            v = votes[winner]
            # Correlated burst observations do NOT amplify score.
            if len(v) == len(inds):
                score = float(np.median([x["score"] for x in v]))
                suggestion = {
                    "cluster": winner,
                    "name": v[0]["best"]["name"],
                    "score": score,
                    "status": "suggested-only",
                }
        out.append(
            {
                "proposal_key": hashlib.sha256(
                    ",".join(map(str, ids)).encode()
                ).hexdigest()[:16],
                "face_ids": ids,
                "max_distance": float(distances(mat[inds], mat[inds]).max()),
                "suggested_identity": suggestion,
                "action": "create-unnamed-candidate",
            }
        )
    return sorted(out, key=lambda x: x["face_ids"])


def main(input_path, outpath, confirmed=None):
    confirmed = confirmed or {}
    out = pathlib.Path(outpath)
    out.mkdir(parents=True, exist_ok=True)
    data = (
        input_path
        if isinstance(input_path, dict)
        else json.loads(pathlib.Path(input_path).read_text(encoding="utf-8"))
    )
    rows = [r for r in data["faces"] if valid(r)]
    if len({r["user_id"] for r in rows}) != 1:
        raise ValueError("One user per job; never mix users")
    rows.sort(key=lambda r: r["id"])
    assign_content_groups(rows)
    refs, partitions, case_groups = make_split(rows, confirmed)
    gallery = Gallery(refs)
    print(
        json.dumps(
            {
                "phase": "validation-gallery",
                "references": len(gallery.rows),
                "conflicts": len(gallery.conflicts),
            }
        ),
        flush=True,
    )
    evaluated = {}
    for split, queries in partitions.items():
        rs = []
        for q, c in queries:
            r = gallery.query(q, exclude_group=True)
            if r:
                rs.append(
                    dict(
                        r,
                        id=q["id"],
                        expected=c,
                        correct=int(
                            r["best"]["cluster"] == c
                            and bool(r["best"]["name"].strip())
                        ),
                    )
                )
        evaluated[split] = rs
    train = evaluated["train"]
    if not train or len({r["correct"] for r in train}) < 2:
        raise ValueError(
            "Training needs both correct and incorrect examples in disjoint groups"
        )
    if not evaluated["calibration"] or not evaluated["test"]:
        raise ValueError("Not enough disjoint calibration and test groups")
    X = np.array([r["features"] for r in train])
    y = np.array([r["correct"] for r in train])
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(C=0.5, max_iter=2000, random_state=42).fit(
        scaler.transform(X), y
    )

    # Scores are model estimates, not claimed calibrated identity probabilities.
    def score(r):
        return float(clf.predict_proba(scaler.transform([r["features"]]))[0, 1])

    for rs in evaluated.values():
        for r in rs:
            r["score"] = score(r)
    cal = evaluated["calibration"]
    choices = []
    for cutoff in np.arange(0.50, 0.996, 0.005):
        m = evaluate(cal, float(cutoff))
        # Provisional 2% wrong among accepted + 2% unknown acceptance ceiling.
        if (
            m.get("accepted", 0) >= 10
            and m.get("wrong", 0) / m["accepted"] <= 0.02
            and m.get("unknown_false_accept", 0) / max(1, m["unknown"]) <= 0.02
        ):
            choices.append((m.get("correct", 0), float(cutoff)))
    cutoff = max(choices)[1] if choices else 1.01
    metrics = {s: evaluate(rs, cutoff) for s, rs in evaluated.items()}
    test = metrics["test"]
    gate = bool(
        choices
        and test.get("accepted", 0) >= 20
        and test["unknown"] >= 20
        and test.get("wrong", 0) / max(1, test.get("accepted", 0)) <= 0.02
        and test.get("unknown_false_accept", 0) / max(1, test["unknown"]) <= 0.02
    )
    (out / "model.json").write_text(
        json.dumps(
            {
                "version": VERSION,
                "features": FEATURES,
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(),
                "coef": clf.coef_.tolist(),
                "intercept": clf.intercept_.tolist(),
                "cutoff": cutoff,
                "calibration_method": "regularized logistic fit; threshold selected on disjoint calibration groups; score is not a calibrated probability",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    report = {
        "version": VERSION,
        "mode": "shadow",
        "validation_gate": gate,
        "cutoff": cutoff,
        "metrics": metrics,
        "limitations": [
            "Existing named labels are provisional, not all manually audited.",
            "Group split uses content and near-duplicate proxies, not verified shooting sessions.",
            "Final gallery is larger than validation gallery; shadow proposals require review.",
            "No photo-quality features until trustworthy measurements exist.",
            "No calibrated probability claim.",
        ],
        "split_ids": {s: [r["id"] for r in rs] for s, rs in evaluated.items()},
        "gallery_ids": [r["id"] for r in gallery.rows],
    }
    (out / "validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "phase": "validated",
                "gate": gate,
                "cutoff": cutoff,
                "metrics": metrics,
            }
        ),
        flush=True,
    )
