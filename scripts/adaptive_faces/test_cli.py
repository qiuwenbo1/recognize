# SPDX-FileCopyrightText: 2026 Recognize contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from adaptive import FEATURES, VERSION, Gallery, make_split, order, valid
from cli import load_model, load_snapshot, predict, propose
from test_adaptive import row


class CliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = dict(
            version=VERSION,
            features=FEATURES,
            embedding_model="synthetic-unit-v1",
            mean=[0] * 5,
            scale=[1] * 5,
            coef=[[-2, 1, 1, 0, 0]],
            intercept=[1],
            cutoff=0.6,
        )

    def save(self, name, data):
        path = self.root / name
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_empty_gallery_has_no_match(self):
        self.assertIsNone(Gallery([]).query(row(1, 0)))

    def test_one_identity_cannot_make_open_set_decision(self):
        self.assertIsNone(Gallery([row(1, 0.1, 1)]).query(row(2, 0)))

    def test_malformed_and_nonfinite_vectors_rejected(self):
        for vector in [
            None,
            ["bad"] * 512,
            [float("nan")] * 512,
            [float("inf")] * 512,
            [0] * 512,
        ]:
            self.assertFalse(valid({"vector": vector}))

    def test_scoring_model_identity_and_shapes_checked(self):
        path = self.save("model.json", self.model)
        self.assertEqual(load_model(path, "synthetic-unit-v1"), self.model)
        with self.assertRaises(ValueError):
            load_model(path, "other-model")
        for field, value in [
            ("scale", [0] * 5),
            ("coef", [[1]]),
            ("cutoff", float("nan")),
            ("features", FEATURES[::-1]),
        ]:
            with self.assertRaises(ValueError):
                load_model(
                    self.save("bad.json", dict(self.model, **{field: value})),
                    "synthetic-unit-v1",
                )

    def test_frozen_score_matches_sklearn(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(4)
        x = rng.normal(size=(100, 5))
        y = (x[:, 0] > 0).astype(int)
        scaler = StandardScaler().fit(x)
        classifier = LogisticRegression(C=0.5).fit(scaler.transform(x), y)
        model = dict(
            self.model,
            mean=scaler.mean_.tolist(),
            scale=scaler.scale_.tolist(),
            coef=classifier.coef_.tolist(),
            intercept=classifier.intercept_.tolist(),
        )
        for features in rng.normal(size=(10, 5)):
            self.assertAlmostEqual(
                predict({"features": features}, model),
                classifier.predict_proba(scaler.transform([features]))[0, 1],
                places=14,
            )

    def test_snapshots_reject_mixed_users_and_duplicate_ids(self):
        r = dict(row(1, 0), user_id="one")
        base = dict(
            schema="recognize-adaptive-snapshot-v1", embedding_model="synthetic-unit-v1"
        )
        for faces in [[], [r, r], [r, dict(r, id=2, user_id="two")]]:
            with self.assertRaises(ValueError):
                load_snapshot(self.save("input.json", dict(base, faces=faces)))

    def test_no_automatic_reference_feedback(self):
        rows = [row(1, 0.1, 1), row(2, 0.2, 1), row(3, 1.8, 2), row(4, 0)]
        rows[0]["reference_eligible"] = False
        propose({"faces": rows}, self.model, self.root)
        decision = json.loads(
            (self.root / "suggestions.jsonl").read_text(encoding="utf-8").strip()
        )
        self.assertEqual(decision["best"]["reference"], 2)
        self.assertFalse(decision["write_authorized"])
        self.assertEqual(rows[3]["cluster_id"], None)

    def test_group_photos_do_not_leak_unknown_identities_between_splits(self):
        labels = sorted(range(1, 31), key=order)
        unknown, known = labels[0], labels[1]
        rows = [
            row(c * 100 + i, i * 0.01, c, group=c * 100 + i)
            for c in labels
            for i in range(25)
        ]
        # One photo contains both a withheld and a known person.
        rows.extend(
            [row(90001, 0.2, unknown, group=777), row(90002, 0.3, known, group=777)]
        )
        refs, parts, _ = make_split(rows)
        seen = {r["content_group"]: "gallery" for r in refs}
        for split, queries in parts.items():
            for r, expected in queries:
                self.assertIn(seen.get(r["content_group"]), [None, split])
                seen[r["content_group"]] = split

    def test_train_then_propose_end_to_end(self):
        rng = np.random.default_rng(10)
        faces = []
        for person in range(1, 41):
            for group in range(10):
                base = np.zeros(512)
                base[person] = 1
                base[50 + group] = 0.5
                for duplicate in range(2):
                    v = base + rng.normal(scale=0.001, size=512)
                    v /= np.linalg.norm(v)
                    i = len(faces) + 1
                    r = dict(
                        row(i, 0, person, title="synthetic-" + str(person)),
                        user_id="synthetic",
                        vector=v.tolist(),
                        vector_hash="synthetic",
                    )
                    faces.append(r)
        query = dict(
            faces[0],
            id=9999,
            file_id=9999,
            cluster_id=None,
            title=None,
            content_key="file:9999",
        )
        faces.append(query)
        snapshot = self.save(
            "snapshot.json",
            dict(
                schema="recognize-adaptive-snapshot-v1",
                embedding_model="synthetic-unit-v1",
                faces=faces,
            ),
        )
        script = str(Path(__file__).with_name("cli.py"))
        training = self.root / "trained"
        proposals = self.root / "proposals"
        subprocess.run(
            [
                sys.executable,
                script,
                "train",
                "--input",
                str(snapshot),
                "--out",
                str(training),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                sys.executable,
                script,
                "propose",
                "--input",
                str(snapshot),
                "--out",
                str(proposals),
                "--model",
                str(training / "model.json"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            json.loads((proposals / "summary.json").read_text())["database_writes"], 0
        )
        self.assertEqual(
            json.loads(snapshot.read_text())["faces"][-1]["cluster_id"], None
        )
        self.assertTrue((training / "validation.json").is_file())
        result = subprocess.run(
            [
                sys.executable,
                script,
                "propose",
                "--input",
                str(snapshot),
                "--out",
                str(proposals),
                "--model",
                str(training / "model.json"),
            ],
            capture_output=True,
        )
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
