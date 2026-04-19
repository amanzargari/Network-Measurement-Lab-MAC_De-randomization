# MAC Address De-randomization via Online Clustering of Probe Requests

Homework #3 for the **Network Measurement and Data Analysis Lab** (Politecnico di Milano — Redondi & Musumeci). The goal is to defeat MAC randomization: given a stream of Wi-Fi probe requests where the source MAC address is continuously re-randomized, recover how many physical devices produced them by fingerprinting the Information Elements (IEs) of each probe.

## Problem

Modern smartphones transmit probe requests with locally-administered randomized MAC addresses (detectable by the 2/6/A/E digit in the second nibble of the first octet). This breaks trivial device counting. However, each hardware/software stack still emits characteristic IEs — HT Capabilities, Extended Capabilities, Vendor Specific Tags, Supported Rates, and frame Length — that act as a stable fingerprint. Clustering bursts of probe requests by IE similarity should recover one cluster per physical device.

## Approach

Pipeline:

1. **Load & merge** the 7 ground-truth CSVs in `Data/MAC_derand_lecture-dataset/lecture-dataset/`.
2. **Drop sparse IEs** — any column with more than 60% missing values is removed (`VHT Capabilities`, `SSID`, `HE Capabilities`).
3. **Label-encode** the remaining categorical IEs; keep `Length`, `Channel`, `DS Channel` numeric.
4. **Burst generation**: group consecutive probes with the same MAC, split a new burst whenever the inter-probe gap exceeds 2.0 s. Each burst is represented by the feature vector of its first probe.
5. **Feature importance**: Mutual Information against the device label ranks the IEs by predictive power (used to sanity-check the chosen `N`).
6. **Online clustering**: for every burst, in arrival order, either
   - assign it to the first existing cluster whose center matches at least `N` features, or
   - open a new cluster with this burst as the center.
7. **Hyperparameter sweep**: run the clusterer for `N = 1 … max_features` and pick the `N` that jointly maximizes V-Measure and minimizes |predicted − actual| device count.
8. **Validation** on `Data/MAC_derand_challenge-dataset/challenge-dataset/` (6 unseen devices): for each test-set size `K ∈ {2, 3, 4, 5}` draw 5 random K-device subsets, for `K = 6` one subset. `random.seed(42)` is fixed for reproducibility. Report per-trial and average Homogeneity, Completeness, V-Measure, and Error. Plot averages vs K.
9. **Unlabelled estimation** on `Data/MAC_derand_unlabelled-challenge.csv` using the best `N`.

## Results

### Hyperparameter sweep (lecture set, 7 ground-truth devices)

| N | Clusters found | Error | Homogeneity | Completeness | V-Measure |
|---|---:|---:|---:|---:|---:|
| 1 | 2  | 5 | 0.38 | 0.98 | 0.54 |
| 2 | 3  | 4 | 0.32 | 0.90 | 0.47 |
| 3 | 7  | 0 | 0.64 | 0.92 | 0.75 |
| **4** | **7**  | **0** | **0.73** | **0.99** | **0.84** |
| 5 | 9  | 2 | 0.73 | 0.96 | 0.83 |
| 6 | 16 | 9 | 1.00 | 0.88 | 0.94 |
| 7 | 73 | 66 | 1.00 | 0.41 | 0.58 |

`N = 4` is the sweet spot: the algorithm recovers exactly 7 clusters while keeping a strong V-Measure. Low `N` under-clusters (high completeness, poor purity); high `N` over-clusters (perfect purity, completeness collapses).

### Validation on challenge devices (N = 4, seed = 42)

| K | Homogeneity | Completeness | V-Measure | Error |
|---:|---:|---:|---:|---:|
| 2 | 1.00 | 1.00 | 1.00 | 0.0 |
| 3 | 0.89 | 0.98 | 0.92 | 0.0 |
| 4 | 0.93 | 0.99 | 0.96 | 0.0 |
| 5 | 0.92 | 0.99 | 0.96 | 0.0 |
| 6 | 0.93 | 0.99 | 0.96 | 0.0 |

Error is 0 at every K — the correct device count is recovered in every single trial. V-Measure is perfect at K=2 across all 5 sampled pairs, dips to 0.92 at K=3 (driven by a single outlier trial: one triplet of devices with heavily overlapping IEs collapsed to V=0.61 while the other four trials achieved V=1.00), and stabilises around 0.96 for K ≥ 4. Completeness stays consistently higher than Homogeneity, meaning the residual errors come from the algorithm splitting one device across two clusters rather than merging two devices into one — the conservative failure mode, preferable for de-randomization.

### Unlabelled capture

Running the best configuration on `MAC_derand_unlabelled-challenge.csv` (20,464 probes → 2,455 bursts after MAC + 2 s gap grouping) yields an estimate of **8 distinct devices** in the monitored environment. The cluster-size distribution is long-tailed — two dominant clusters (1260 and 1084 bursts) account for most traffic, likely stationary devices, while several clusters contain only 1–2 bursts, likely transient devices passing briefly through the sensor's range.

## Repository layout

```
.
├── NML-HW3.ipynb                                # Full pipeline
├── Data/
│   ├── MAC_derand_lecture-dataset/
│   │   └── lecture-dataset/                     # 7 labelled CSVs for N selection
│   ├── MAC_derand_challenge-dataset/
│   │   └── challenge-dataset/                   # 6 labelled CSVs for validation
│   └── MAC_derand_unlabelled-challenge.csv      # unknown environment
├── LICENSE
└── README.md
```

## Running it

```bash
pip install pandas numpy scikit-learn matplotlib seaborn tqdm jupyter
```

Run cells top to bottom. The whole pipeline takes a few minutes on a laptop (the online clustering is O(bursts × clusters × features) and the lecture set is under 1k bursts).

## Design notes and limitations

- **Burst representation.** Each burst is represented by the feature vector of its first probe. Centers are fixed at cluster creation and never updated; an alternative would be to update the center with a per-feature majority vote after each assignment.
- **Tie-breaking.** When several clusters satisfy the `≥ N` matches rule, the first one found wins (as the assignment allows).
- **Dropped IEs.** `SSID` is dropped because 91% of probes omit it (modern Android/iOS randomization actively hides it). `VHT/HE Capabilities` are dropped because they are essentially never populated in this capture.
- **Label encoding is per-dataset.** Encoders are refit on each dataset, so numeric codes are not comparable across lecture / challenge / unlabelled — this is fine because the clustering is online and each run is self-contained.
- **Sensitivity to device combinations.** As the K=3 outlier trial shows, performance depends not only on *how many* devices are in the environment but *which* specific devices: two devices with near-identical IE signatures can collapse into one cluster even at the optimal N.
- **Reproducibility.** `random.seed(42)` is fixed at the start of the notebook so the K-device subsets in the validation loop are identical across runs.

## References

- Slides: *MAC randomization / MAC de-randomization*, Network Measurement and Data Analysis Lab, Politecnico di Milano.
- Homogeneity, Completeness, V-Measure: Rosenberg & Hirschberg, *EMNLP-CoNLL 2007*.
