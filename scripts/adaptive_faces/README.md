# Experimental adaptive face proposals

This opt-in tool evaluates named-person matches and small unnamed groups from a
single user's existing embeddings. It does not replace Recognize's background
jobs, generate embeddings, migrate models, or modify any face assignments.
Python dependencies are installed separately; normal Recognize installations do
not need them. No trained scoring model or personal dataset is distributed.

The motivation is that a single nearest-distance threshold can reject difficult
photos of a well-represented person, while a more permissive threshold accepts
lookalikes. This experiment also considers the nearest competing identity,
the reference's nearest different-identity distance, and independent support.

## Input and model compatibility

Only finite, L2-normalized 512-component descriptors are supported. The legacy
128-component descriptors are excluded, never padded or mixed into this space.
Two models with 512 components are not necessarily compatible. All vectors in an
input must come from the same weights and preprocessing pipeline. The exporter
requires an administrator-supplied model identifier; it cannot independently
verify how existing database embeddings were generated. Frozen scorers reject a
different model identifier. This is not an InsightFace installation or migration
tool.

## Run locally on the Nextcloud server

Use a private directory outside the web root and repository. The export contains
biometric descriptors, user IDs and person names. Do not attach it, the trained
model, or the generated reports to a public issue/PR. Unix examples:

```sh
umask 077
php occ recognize:adaptive:export USER MODEL-AND-PREPROCESSING-REVISION > /private/faces.json
python3 -m venv /private/adaptive-venv
/private/adaptive-venv/bin/pip install -r scripts/adaptive_faces/requirements.txt
/private/adaptive-venv/bin/python scripts/adaptive_faces/cli.py train \
  --input /private/faces.json --out /private/training-run
/private/adaptive-venv/bin/python scripts/adaptive_faces/cli.py propose \
  --input /private/faces.json --model /private/training-run/model.json \
  --out /private/proposal-run
```

Run `occ` as the Nextcloud service account. On Windows, restrict the export and
output directories using filesystem ACLs; `umask` alone does not provide that
protection. Each output directory must be new. Training writes `model.json` and
`validation.json`. Proposal generation writes `suggestions.jsonl`, `clusters.json`,
`conflicts.json` and `summary.json`; each run records the input hash in a manifest.
No command provides `--apply`.

The JSON envelope is `schema: recognize-adaptive-snapshot-v1`,
`embedding_model: <identifier>`, and `faces: [...]`. Each row contains `id`,
`user_id`, `file_id`, `cluster_id`, `threshold`, `x`, `y`, `width`, `height`,
`vector`, `vector_hash`, `title`, and `content_key`. The export defaults content
keys to file IDs. An offline preparation step can supply an image-content hash
to detect the same photo stored as multiple files. Optional
`reference_eligible: false` prevents a previously automated assignment from
becoming a reference or training observation. The export has no source-history
metadata, so existing labels are provisional references by default.

## Matching and clustering

* Query faces must be unassigned, have no positive manual-removal threshold,
  and be at least 0.03 of the image width and height. Assigned faces are never
  relocation candidates.
* References identical under contradictory labels are removed. Self, same-file,
  same-content and exact-descriptor matches cannot vote for the query.
* All eligible references participate. Unnamed groups compete with named people,
  but an unnamed winner cannot generate a named assignment suggestion.
* A regularized logistic scorer uses nearest distance, competitor gap, local
  different-identity gap, and log-scaled support/reference counts. Its output is
  a ranking score, not a calibrated probability of identity.
* Support is limited to independent content/near-duplicate groups. Distance 0.35
  defines the near-duplicate proxy; up to five groups within 0.10 of the nearest
  reference contribute support. These are experimental assumptions to validate
  for each model, not universal face-recognition thresholds.
* Remaining pending faces use global mutual 20-nearest-neighbor edges and a
  complete-link diameter limit of 0.70. A group needs three detections on distinct
  files. The diameter check prevents weak transitive chains joining distant
  endpoints. It can split a real person across poses; three photos alone do not
  guarantee a group. These constants are independent of batch size.

## Validation and limitations

Training, cutoff selection and held-out evaluation use disjoint content groups.
Whole named identities are withheld to measure unknown-person acceptance. Group
photos shared by withheld and known identities stay in one partition; ambiguous
cross-partition groups are excluded. These groups are leakage proxies, not verified
capture sessions. Existing labeling errors affect both fitting and the reported
error counts. The final gallery is larger than the validation gallery, which is
another source of distribution shift.

The provisional calibration rule requires at least ten accepted examples and at
most 2% wrong among accepted and 2% acceptance of withheld identities. The test
gate also requires at least twenty accepted and twenty unknown examples. These
are empirical checks on that split, not statistical confidence bounds. If no
cutoff qualifies, a cutoff above 1 suppresses all named suggestions. Sparse or
single-class training data produces an explicit error. A failed held-out gate
is reported; the proposal command remains a read-only inspection tool.

This contribution contains the reusable proposal algorithm and export boundary.
Automatic write-back, concurrency checks, durable assignment provenance, model
migration and deployment scheduling need separate upstream design and review.
Any future writer must verify snapshot freshness, preserve manual decisions and
existing assignments, audit each change, and prevent automatic assignments from
silently reinforcing themselves. Do not wire these reports directly to SQL updates.

Run synthetic tests with:

```sh
python -m unittest discover -s scripts/adaptive_faces -p 'test_*.py' -v
```

The PHP export test uses the repository's normal Nextcloud PHPUnit environment:
`composer test:unit -- --filter ExportAdaptiveFacesTest`.
