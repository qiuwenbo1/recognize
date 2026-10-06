# SPDX-License-Identifier: AGPL-3.0-or-later
import unittest
import json
import numpy as np
from adaptive import Gallery, build_clusters, eligible, make_split


def row(i, angle, cluster=None, file=None, group=None, title="person", threshold=0):
    v = np.zeros(512)
    v[0] = np.cos(angle)
    v[1] = np.sin(angle)
    return dict(
        id=i,
        file_id=file or i,
        content_key=str(file or i),
        content_group=group or i,
        vector=v.tolist(),
        cluster_id=cluster,
        title=title,
        threshold=threshold,
        width=0.2,
        height=0.2,
    )


class AdaptiveTests(unittest.TestCase):
    def test_conflicting_labels_cannot_vote(self):
        g = Gallery([row(1, 0.1, 1), row(2, 0.1, 2), row(3, 0.2, 1), row(4, 1.4, 2)])
        self.assertEqual(g.conflicts, [[1, 2]])
        self.assertEqual(g.query(row(9, 0))["best"]["reference"], 3)

    def test_same_photo_and_group_are_excluded(self):
        g = Gallery(
            [
                row(1, 0.01, 1, file=9),
                row(2, 0.1, 1, group=9),
                row(3, 0.4, 1),
                row(4, 1.4, 2),
            ]
        )
        q = row(9, 0)
        self.assertEqual(g.query(q, True)["best"]["reference"], 3)

    def test_three_close_faces_form_cluster(self):
        rs = [row(1, 0), row(2, 0.1), row(3, 0.2), row(4, 2), row(5, 2.1), row(6, 2.2)]
        cs = build_clusters(rs, {}, 0.8)
        self.assertEqual([c["face_ids"] for c in cs], [[1, 2, 3], [4, 5, 6]])

    def test_single_link_chain_cannot_merge_distant_endpoints(self):
        rs = [row(i + 1, a) for i, a in enumerate([0, 0.1, 0.2, 0.6, 1, 1.1, 1.2])]
        for c in build_clusters(rs, {}, 0.8):
            self.assertLessEqual(c["max_distance"], 0.7 + 1e-9)

    def test_same_photo_not_merged(self):
        self.assertEqual(
            build_clusters(
                [row(1, 0, file=7), row(2, 0.1, file=7), row(3, 0.2)], {}, 0.8
            ),
            [],
        )

    def test_holds_assigned_invalid_and_small_excluded(self):
        self.assertFalse(eligible(row(1, 0, threshold=0.1)))
        self.assertFalse(eligible(row(1, 0, cluster=2)))
        r = row(1, 0)
        r["vector"] = [0] * 128
        self.assertFalse(eligible(r))
        r = row(1, 0)
        r["width"] = 0.01
        self.assertFalse(eligible(r))

    def test_deterministic_clusters(self):
        rs = [row(i + 1, i * 0.1) for i in range(4)]
        self.assertEqual(build_clusters(rs, {}, 0.8), build_clusters(rs, {}, 0.8))

    def test_output_json_and_finite_features(self):
        g = Gallery([row(1, 0.2, 1), row(2, 1, 2)])
        result = g.query(row(9, 0))
        json.dumps(result, allow_nan=False)

    def test_split_has_no_content_or_unknown_identity_leakage(self):
        rows = [
            row(20000 + c * 40 + i, i * 0.01, c, group=20000 + c * 40 + i // 2)
            for c in range(1, 31)
            for i in range(30)
        ]
        refs, parts, _ = make_split(rows)
        gallery_groups = {r["content_group"] for r in refs}
        gallery_labels = {r["cluster_id"] for r in refs}
        seen = {}
        for split, queries in parts.items():
            for r, expected in queries:
                self.assertNotIn(r["content_group"], gallery_groups)
                if expected is None:
                    self.assertNotIn(r["cluster_id"], gallery_labels)
                if r["content_group"] in seen:
                    self.assertEqual(seen[r["content_group"]], split)
                seen[r["content_group"]] = split


if __name__ == "__main__":
    unittest.main()
