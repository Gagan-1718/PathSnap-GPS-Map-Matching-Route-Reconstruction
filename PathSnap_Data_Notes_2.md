# PathSnap: Member 3 (Data) — what to push

You push **2 things**:

| Part | Save it in the repo as |
|---|---|
| Part A: Data notes (below) | `data_processing/DATA_NOTES.md` |
| The Colab notebook **after you run it** (with outputs) | `PathSnap_Colab.ipynb` (replace the existing file) |

## Before you start
1. Accept the GitHub collaborator invite from Gagan.
2. Get the code:
   ```
   git clone https://github.com/Gagan-1718/PathSnap-GPS-Map-Matching-Route-Reconstruction.git
   cd PathSnap-GPS-Map-Matching-Route-Reconstruction
   pip install -r requirements.txt
   ```
3. Put the dataset in `data\` as described in `data/README.md`.

## Steps
1. Make your branch: `git switch -c data-notes-and-colab-run`
2. **Check Part A:** open the CSVs it lists and confirm every number. Then rewrite it in your own words, because reports are checked for AI-written text.
3. Save Part A as `data_processing/DATA_NOTES.md`.
4. **Run the notebook:** open
   https://colab.research.google.com/github/Gagan-1718/PathSnap-GPS-Map-Matching-Route-Reconstruction/blob/main/PathSnap_Colab.ipynb
   - Runtime → Run all. Upload `Student folder.zip` when it asks.
   - The KNN and Bayesian-tuning cells are slow. Let them finish.
   - File → Download → `.ipynb`, and replace `PathSnap_Colab.ipynb` in the repo with it.
5. Push (2 commits):
   ```
   git add data_processing/DATA_NOTES.md
   git commit -m "docs(data): data-quality issues, assumptions and decisions"
   git add PathSnap_Colab.ipynb
   git commit -m "feat: executed colab notebook with outputs"
   git push -u origin data-notes-and-colab-run
   ```
6. On GitHub, open a **Pull request** to `main` and tell Gagan.

---

## Part A: `data_processing/DATA_NOTES.md`

# Data notes: quality issues, assumptions and decisions

> **Draft.** Check every number against the files listed before you push, and rewrite it in your own words.
> Sources: `results/eda/trip_summary.csv`, `results/eda/split_summary.csv`, `results/features/*_candidate_stats.csv`.

## 1. What the raw data is

| File | Columns | Meaning |
|---|---|---|
| `.track` | lon, lat, time (s) | GPS points of one trip |
| `.nodes` | lon, lat | road points; row number = node id |
| `.arcs` | from_node, to_node | directed road segments; row number = arc id |
| `.route` | arc_id | reference route: the arcs driven, in order |

- A GPS row is **not** linked to a route row. A trip has many more GPS points than route arcs (train median: 1,529 points vs 280 arcs). So the per-point labels must be derived (section 4).
- Arcs are **directed**. The same street usually has one arc in each direction, so heading is needed to tell them apart.

## 2. Partitions used

| Partition | Trips | GPS points | Used for |
|---|---|---|---|
| train | 80 | 208,417 | fitting, cross-validation, tuning |
| student_test | 10 | 19,160 | comparing models only |
| evaluator_test | 10 | (not inspected) | final demo only; never used for any choice |

## 3. Data-quality checks (train)

| Check | Result | What we did |
|---|---|---|
| Missing values | 0 | nothing needed |
| Duplicate timestamps | 0 | the drop step stays in the code as a safeguard |
| Out-of-order timestamps | 0 | we still sort by time as a safeguard |
| Sampling rate | 93.8% of steps are exactly 1 s apart (test: 98.2%) | none |
| Time gaps over 10 s | in 40 of 80 trips; the largest gap is 153,064 s (about 42 h) | the HMM resets after a gap of more than 10 s |
| Biggest single jump | 887.6 m between two points | points too fast to reach and to leave (over 50 m/s = 180 km/h) are removed as GPS spikes |
| Trip size | 361 to 19,313 points (median 1,529); median length 24.1 km | none |
| Location | worldwide: longitude −100.4 to 137.2, latitude −34.6 to 61.7 | each trip gets its own local projection to metres |
| Road network size | median 9,608 nodes and 19,526 arcs per trip | arcs far from the track are skipped for speed |
| Isolated nodes | 0 | none |
| Disconnected parts | up to 421 components in one trip; the largest component holds at least 69.3% of nodes | the HMM penalises jumps between unconnected arcs instead of forbidding them |
| Reference route connectivity | 99.93% of consecutive route arcs are connected | the route is mostly clean |

## 4. Assumptions and decisions

1. **Distance in metres, not degrees.** One degree of longitude shrinks away from the equator, so we use a local equirectangular projection centred on each trip. It's accurate over a few kilometres.
2. **Candidates.** Up to 10 nearest arcs within 50 m of each point. If none are within 50 m, we widen the search to 200 m.
3. **Labels are derived.** A point's label is the nearest **route** arc within 30 m. If the car's direction is reliable, an arc pointing the wrong way gets a +25 m penalty.
4. **Unlabelled points.** If no route arc is within 30 m, the point is left out of training and scoring. We don't treat the route as perfect ground truth.
5. **Heading when stopped.** If the car moved less than 5 m across neighbouring points, its heading is noise, so the feature is set to a neutral 90° and flagged as unreliable.

## 5. The biggest data issue: the route doesn't always cover the whole track

| Partition | GPS points labelled | Trips with low coverage |
|---|---|---|
| train | 182,079 of 207,454 (87.8%) | trip 00000054: 11.5%; trip 00000033: 29.1% |
| student_test | 19,018 of 19,160 (99.3%) | none |

In trip 54, the GPS track is about 169 km but the reference route is only about 31 km. Before we found this, one cross-validation fold scored 39% top-1 accuracy because uncovered points counted as errors.

## 6. Candidate quality

- Candidate rows: 1,400,327 (train) and 131,630 (student_test).
- **Candidate miss rate** (the labelled road is not in the candidate list): **0.31%** in both partitions.
