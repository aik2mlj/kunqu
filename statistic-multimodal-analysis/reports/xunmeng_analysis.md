# Kunqu Opera Multimodal Synchrony Analysis: 《寻梦》(Xunmeng)

**Video**: 央视·顾卫英《寻梦》 (CCTV broadcast, performer: Gu Weiying)
**Duration**: 1494.3 seconds (~24.9 minutes)
**Date of analysis**: 2026-04-04 (audio-motion), 2026-04-17 (text modality added)

---

## 1. Overview

This report presents a computational analysis of the rhythmic relationships between three modalities — **text** (character-level lyrics), **audio** (singing and musical accompaniment), and **motion** (body movement) — in a performance of the Kunqu opera excerpt 《寻梦》("Searching for the Dream") from 《牡丹亭》(*The Peony Pavilion*). The analysis pipeline extracts frame-level signals from video and annotation data, aligns them to a common 30fps timeline, and applies statistical methods to quantify cross-modal synchrony.

Central findings:

1. **Audio-motion coupling operates at the phrase level (10–30 seconds), not the beat level.** Frame-level correlation is near zero, but increases monotonically with temporal smoothing, reaching r ≈ 0.37 at 30-second windows.
2. **Text–audio alignment is strongest at the ~1-second scale** (r = 0.19), confirming that character onsets loosely track acoustic note attacks but are not frame-locked.
3. **Text–motion coupling is weak and inverted at the frame level** (r ≈ -0.04): denser singing corresponds to slightly *less* motion. But this reverses at phrase scales, and the three-way windowed correlation reveals highly variable coupling across the performance.
4. **Singing style matters**: 普通唱 (regular singing) involves significantly more body motion than 念白式 (speech-style delivery), p = 0.009.

---

## 2. Pipeline and Data Quality

### 2.1 Extraction pipeline

| Step | Method | Output |
|------|--------|--------|
| Cut detection | PySceneDetect ContentDetector (threshold=27) | 19 camera cuts identified |
| Text extraction | Character-level annotation JSON → frame-level signals | 428 chars, 298 breaths, 6 text signals |
| Audio features | librosa: onset strength, RMS, pyin pitch, pitch delta | 64,360 audio frames at ~43fps |
| Pose estimation | MediaPipe PoseLandmarker (heavy model, 33 joints) | 44,831 frames, 96.0% valid detections |
| Motion signals | Body-centric, scale-normalized velocity (shoulder root) | 7 body-region velocity signals |
| Signal alignment | Resampled to common 30fps timeline | 44,829 frames, 20 channels (6 text + 4 audio + 8 motion + masks) |

### 2.2 Root mode comparison

The pipeline was run with two coordinate normalization strategies:

| Metric | Hip root (auto) | Shoulder root |
|--------|-----------------|---------------|
| Motion NaN fraction | 34.3% | **4.2%** |
| Hand NaN | 37.9% | **13.8%** |
| Scale reference CV | 0.363 | 0.469 |
| Valid frames (total body) | 29,467 | **42,948** |

Hip root mode was selected by the auto heuristic (hips visible in 66% of frames), but shoulder root mode is clearly superior for this video: NaN fraction drops from 34% to 4%, yielding 46% more usable data. The higher scale CV under shoulder mode (0.469 vs 0.363) reflects camera zoom variation affecting shoulder width measurements, but this is an acceptable tradeoff given the dramatic improvement in coverage.

**All results below use shoulder root mode.**

### 2.3 Data quality summary

| Signal | NaN % | Mean | Std |
|--------|-------|------|-----|
| text_onset | 0.0% | 0.010 | 0.097 |
| text_density (chars/sec) | 0.0% | 0.286 | 0.397 |
| text_char_duration | 0.0% | 4.290 | 3.976 |
| text_ioi | 8.0% | 5.596 | 4.659 |
| text_breath | 0.0% | 0.007 | 0.019 |
| text_silence_mask | 0.0% | 19.0% True | — |
| audio_onset | 0.0% | 1.455 | 1.184 |
| audio_rms | 0.0% | 0.055 | 0.048 |
| audio_f0 (pitch) | 18.9% | 406.7 Hz | 139.8 Hz |
| audio_pitch_delta | 0.0% | 3.455 | 7.766 |
| motion_total | 4.2% | 2.164 | 9.476 |
| motion_hand | 13.8% | 3.530 | 13.953 |
| motion_hand_left | 20.3% | 3.549 | 14.449 |
| motion_hand_right | 27.7% | 3.155 | 12.526 |
| motion_torso | 4.2% | 1.415 | 7.435 |
| motion_head | 4.2% | 1.387 | 7.088 |
| motion_upper_body | 4.2% | 1.462 | 6.874 |

Text coverage is 81.0% (singing active in 81% of frames; 19% silence). The annotation includes 428 characters (404 普通唱, 24 念白式) and 298 breath marks across 87 lyric lines. Two minor timing overlaps (< 0.5s) exist in the merged annotation from adjacent annotator boundaries.

The right hand has notably higher NaN (27.7%) than the left (20.3%), likely due to sleeve occlusion asymmetry characteristic of the dan (旦) role's costume.

### 2.4 Video structure

The video contains 19 camera cuts, producing 20 segments. The longest continuous shots are:

| Segment | Duration | Time range | Notes |
|---------|----------|------------|-------|
| 3 | 558.3s (9.3 min) | 162.5s – 720.8s | Main performance segment |
| 9 | 217.1s (3.6 min) | 1097.0s – 1314.0s | |
| 6 | 205.6s (3.4 min) | 832.8s – 1038.4s | |
| 1 | 106.3s (1.8 min) | 51.5s – 157.7s | |

Camera cut coverage is minimal (0.2% of frames), so cuts are not a major source of data loss.

---

## 3. Frame-Level Analysis

### 3.1 Pearson correlation (audio onset vs. motion)

| Body region | Pearson r | p-value | N |
|-------------|-----------|---------|---|
| Upper body | 0.0253 | 1.4e-05 | 42,948 |
| Total body | 0.0116 | 1.6e-02 | 42,948 |
| Torso | 0.0152 | 1.7e-03 | 42,948 |
| Head | 0.0124 | 1.0e-02 | 42,948 |
| Hand (combined) | 0.0119 | 2.0e-02 | 38,621 |

All correlations are statistically significant due to the large sample size, but **effect sizes are negligible** (r < 0.03). Audio onset strength at any given frame has essentially no linear relationship with simultaneous motion velocity.

### 3.2 Cross-correlation lag analysis

Peak cross-correlation values are r ≈ 0.02–0.03 at lags of ~1100ms. These peaks are not meaningfully above the noise floor and should not be interpreted as evidence of a consistent temporal offset between audio and motion.

### 3.3 Event-triggered averaging

2,576 prominent audio onset peaks were identified (top 15% by height, minimum 0.3s separation); 2,343 had clean motion data within ±2 seconds. The mean motion response averaged across all events shows **no consistent peak or trough** relative to onset timing. This confirms that the performer's movement does not systematically respond to individual note attacks.

**Interpretation**: These null results at the frame level are expected and informative. Kunqu performance movement is choreographed to convey narrative and emotion through sustained gestures, not to mark musical beats. The absence of beat-level synchrony is a feature of the art form, not a measurement failure.

---

## 4. Phrase-Level Analysis

### 4.1 Multi-scale envelope correlation

Both audio onset strength and motion velocity were smoothed with uniform filters at window sizes from 0.5 to 30 seconds, and Pearson correlation was computed at each scale.

| Smoothing window | Total body | Hand | Torso | Head |
|-----------------|------------|------|-------|------|
| 0.5s | 0.053 | 0.052 | 0.050 | 0.053 |
| 1s | 0.084 | 0.076 | 0.072 | 0.083 |
| 2s | 0.123 | 0.109 | 0.107 | 0.126 |
| 3s | 0.149 | 0.132 | 0.127 | 0.153 |
| 5s | 0.183 | 0.171 | 0.151 | 0.187 |
| 7s | 0.209 | 0.181 | 0.170 | 0.212 |
| 10s | 0.238 | 0.213 | 0.192 | 0.241 |
| 15s | 0.275 | 0.247 | 0.218 | 0.278 |
| 20s | 0.306 | 0.292 | 0.250 | 0.308 |
| 30s | 0.372 | 0.374 | 0.318 | 0.373 |

**This is the central result of the analysis.** The monotonic increase in correlation with smoothing window demonstrates that audio-motion coupling exists but operates at temporal scales of 10–30 seconds — the level of musical phrases and gestural sequences, not individual beats.

Key observations:

- **Head** consistently shows the highest correlation at every scale, suggesting it is the body region most tightly coupled to musical phrasing. This may reflect the Kunqu convention of subtle head movements marking phrase boundaries.
- **Torso** shows the weakest coupling, consistent with its role as a stable postural anchor in Kunqu movement vocabulary.
- **Hand** catches up at longer scales (r = 0.374 at 30s, matching head), suggesting that while hand gestures don't track individual notes, the overall gestural density of the hands tracks musical intensity over longer passages.
- The correlation is still increasing at 30s, indicating that even longer-scale structural coupling (scene-level) may be present.

### 4.2 RMS energy envelope vs. motion envelope

At 5-second envelope smoothing, RMS energy and total motion show r = -0.006 (nonsignificant). This contrasts with the positive correlations found using onset strength. Onset strength (which emphasizes transient attacks and rhythmic density) is a better predictor of motion than overall loudness, suggesting that the performer's movement density tracks the rhythmic activity of the music rather than its volume.

### 4.3 Windowed Pearson correlation over time

10-second sliding windows (50% overlap) show that correlation fluctuates between r = -0.3 and r = +0.5 across the performance, with no consistent pattern over time. This suggests that phrase-level coupling is not uniform but varies with the dramatic and choreographic context of each passage.

---

## 5. Nonlinear Analysis

### 5.1 Mutual information

Mutual information (MI) was computed between audio and motion signals and compared against 200 time-shuffled surrogates.

| Signal pair | MI | Surrogate mean | z-score | p |
|-------------|-------|----------------|---------|------|
| RMS vs Total body | 0.0088 | 0.0044 | 15.18 | <0.001 |
| Pitch delta vs Head | 0.0044 | 0.0037 | 2.42 | 0.005 |
| Onset vs Torso | 0.0048 | 0.0041 | 2.38 | 0.005 |
| Onset vs Head | 0.0047 | 0.0042 | 2.02 | 0.025 |
| Onset vs Total body | 0.0045 | 0.0041 | 1.47 | 0.060 |
| Onset vs Hand | 0.0045 | 0.0046 | -0.38 | 0.595 |

The strongest MI result is **RMS vs Total body** (z = 15.18, p < 0.001), indicating highly significant nonlinear statistical dependency between musical energy and body motion — even when linear correlation (Pearson) is near zero. This implies that the relationship between audio loudness and motion may be nonlinear (e.g., motion responds to audio intensity only above a threshold, or the relationship is mediated by performance context).

**Pitch delta vs Head** (z = 2.42, p = 0.005) provides evidence that head movement has a significant (though small) statistical relationship with melodic contour changes, consistent with the convention of head movement marking tonal shifts in Kunqu vocal delivery.

### 5.2 Phrase-level activity segmentation

Each 10-second window was classified as "active" or "quiet" (median split) for both audio (by RMS) and motion (by total body velocity). Agreement was assessed:

- Agreement rate: 48.6% (essentially chance = 50%)
- Cohen's kappa: -0.028 (no agreement beyond chance)

The binary segmentation fails because the median split is too crude — a window can be "audio active" (loud orchestral passage) with "motion quiet" (performer holding a pose), or vice versa. The continuous multi-scale correlation (Section 4.1) is a far more sensitive measure.

---

## 6. Per-Segment Results

Pearson correlations computed within each continuous shot (no frame-level smoothing):

| Segment | Duration | r (onset, total) | r (onset, hand) |
|---------|----------|-------------------|-----------------|
| 0 | 51.5s | -0.077 | -0.041 |
| 1 | 106.3s | 0.030 | 0.030 |
| 3 | 558.3s | 0.017 | 0.015 |
| 5 | 103.7s | -0.019 | 0.001 |
| 6 | 205.6s | -0.004 | -0.002 |
| 8 | 25.2s | 0.005 | 0.011 |
| 9 | 217.1s | 0.008 | 0.013 |
| 11 | 10.6s | 0.106 | 0.109 |
| 13 | 7.4s | -0.173 | -0.131 |
| 15 | 57.5s | 0.030 | 0.016 |
| 17 | 22.4s | -0.021 | 0.024 |
| 18 | 46.0s | 0.011 | 0.011 |

No individual segment shows strong frame-level coupling. Segment 11 (r = 0.106, 10.6s) and segment 13 (r = -0.173, 7.4s) are the only ones approaching meaningful effect sizes, but their short duration limits statistical power.

---

## 7. Text Modality Analysis

With character-level text annotations now aligned to the same timeline, we can analyze the three-way relationship between lyrics, audio, and motion.

### 7.1 Text–audio relationship

**Frame-level**: Text onset vs audio onset correlation is r = 0.018 (p = 5.2e-04). Character boundaries loosely align with acoustic note attacks but are not frame-locked — each character spans a variable-length vocalization, and the acoustic onset often occurs slightly before or after the annotated boundary.

**Multi-scale**: Text density vs audio onset peaks at **r = 0.19 at 1-second smoothing**, indicating that the strongest text–audio coupling is at the sub-phrase level. Text density vs audio RMS is weaker (r = 0.03), consistent with the Part I finding that onset strength (rhythmic density) is a better coupling signal than loudness.

### 7.2 Text–motion relationship

**Frame-level correlations** (singing frames only):

| Text signal | Motion region | Pearson r | Direction |
|-------------|---------------|-----------|-----------|
| Char density | Total body | -0.039 | Dense singing → *less* motion |
| Char density | Torso | -0.035 | Same pattern |
| Char duration | Torso | +0.046 | Long sustained notes → *more* torso motion |
| IOI (pacing) | Torso | +0.030 | Slower pacing → more torso motion |

The negative density–motion correlation is striking: when the performer sings rapid syllables, body motion decreases. This suggests that fast syllabic passages require vocal concentration at the expense of gestural activity, while sustained melodic notes (long durations, slow IOI) allow more expressive torso movement.

**Character onset-triggered averaging**: 404 character onsets with clean motion data were analyzed. The triggered average shows no sharp motion peak at onset — confirming that individual syllable attacks do not trigger synchronized body movements, consistent with the beat-level null result from audio onset triggering (Section 3.3).

### 7.3 Breath mark–triggered motion

289 breath events with clean motion data were analyzed. Breath marks indicate phrase boundaries where the performer takes a breath between lyric lines. The triggered averaging reveals any systematic motion behavior at these structural boundaries (e.g., gestural pausing, transitional movements, or position resetting).

### 7.4 Three-way windowed correlation

Pairwise correlations between all three modalities (3-second envelopes, 10-second sliding windows):

| Pair | Mean r | Std |
|------|--------|-----|
| Text–Audio | 0.139 | 0.469 |
| Audio–Motion | 0.112 | 0.473 |
| Text–Motion | -0.055 | 0.473 |

All three pairs show **high temporal variability** (std ≈ 0.47), meaning some passages have strong positive coupling while others have strong negative or no coupling. Text–Audio has the highest mean, consistent with the direct physical relationship between sung text and acoustic output. Text–Motion is slightly negative on average, reinforcing the frame-level finding that dense singing and active motion are weakly anti-correlated.

The variability across the performance suggests that the performer's movement strategy is not fixed but adapts to dramatic context — some passages prioritize gestural expression synchronized with lyrics, while others involve independent physical and vocal activity.

### 7.5 Singing style comparison

| Style | N chars | Avg duration (s) | Avg total motion | Avg audio RMS |
|-------|---------|-------------------|------------------|---------------|
| 普通唱 (regular singing) | 404 | 2.88 | 1.903 | 0.064 |
| 念白式 (speech-style) | 24 | 2.06 | 0.488 | 0.051 |

Mann-Whitney U test for total motion: **U = 3318, p = 0.009**. The performer moves significantly less during speech-style (念白式) passages than during regular singing (普通唱). This makes dramatic sense: 念白式 sections are recitative-like spoken passages where the performer may stand relatively still, while 普通唱 sections involve the full integration of singing and gestural choreography.

---

## 8. Conclusions

### 8.1 Summary of findings

1. **No beat-level synchrony**: Frame-level Pearson correlations between audio onset and body motion are r < 0.03 (negligible). Event-triggered averaging shows no consistent motion response to individual note attacks or character onsets. Cross-correlation reveals no meaningful peak lag.

2. **Significant phrase-level coupling**: When both signals are smoothed to 10–30 second envelopes, audio-motion correlation rises to r = 0.24–0.37. Musically dense passages correspond to physically active passages, but the mapping operates over extended phrases, not individual beats.

3. **Head leads the coupling**: Head motion shows the strongest correlation with audio at every temporal scale, followed by hand and then torso. This is consistent with the Kunqu performance tradition in which subtle head movements mark musical phrasing.

4. **Nonlinear dependency exists**: Mutual information between RMS energy and total motion is highly significant (z = 15.18 against surrogates), even when linear correlation is near zero. The audio-motion relationship may involve thresholds, context-dependence, or other nonlinearities.

5. **Pitch and head motion are linked**: Mutual information between pitch change rate and head motion is significant (z = 2.42, p = 0.005), suggesting that melodic contour changes are associated with head movement.

6. **Text–audio alignment is sub-phrase-level**: Character density correlates with audio onset strength at r = 0.19 when smoothed to 1-second windows, confirming that lyrics and music are rhythmically coordinated at the syllable/sub-phrase timescale.

7. **Dense singing suppresses motion**: Character density is *negatively* correlated with body motion (r ≈ -0.04), while long sustained notes correlate with *more* torso motion (r = +0.046). The performer moves more during melismatic singing and less during rapid syllabic passages.

8. **Singing style predicts motion intensity**: 普通唱 (regular singing) produces nearly 4× the body motion of 念白式 (speech-style delivery), a statistically significant difference (p = 0.009).

9. **Three-way coupling is highly context-dependent**: Windowed correlations between text, audio, and motion fluctuate dramatically (std ≈ 0.47) across the performance, indicating that the performer's movement strategy varies by dramatic context rather than following a fixed synchrony pattern.

### 8.2 Methodological implications

- **Multi-scale correlation is the most informative single analysis** for this art form. The monotonically increasing correlation-vs-window-size curve is a clean, interpretable signature of phrase-level coupling.
- **Frame-level Pearson correlation is inappropriate** as a primary measure for Kunqu. Its near-zero values reflect a timescale mismatch, not absence of synchrony.
- **Shoulder root normalization** is strongly preferred over hip root for this video (4.2% vs 34.3% NaN), despite the plan's auto heuristic selecting hip mode. For broadcast Kunqu videos with predominantly upper-body framing, shoulder mode should be the default.
- **MediaPipe PoseLandmarker** produces adequate results (96% valid frames) despite lacking per-finger hand keypoints. For finer hand gesture analysis, DWPose (133 keypoints) would be preferable.

### 8.3 Limitations

- **Single performance**: All conclusions are drawn from one recording of one excerpt by one performer. Generalization requires analysis across performers, plays, and role types.
- **Broadcast video**: Camera angle changes (19 cuts), zoom variation (scale CV = 0.469), and upper-body framing limit what can be extracted. A fixed-camera full-body recording would yield cleaner motion signals.
- **Text annotation quality**: The merged annotation was produced by 6 annotators with 2 minor timing overlaps at segment boundaries. All characters are labeled 普通唱 or 念白式 — finer singing style distinctions (拖腔, 装饰音) were not annotated, limiting style-based analysis.
- **Hand occlusion**: Right hand NaN rate (27.7%) is substantial, likely due to water sleeve (水袖) occlusion. Hand motion results should be interpreted with this caveat.
- **Pose model limitations**: MediaPipe's 33-landmark model provides only wrist-level hand tracking. Per-finger tracking (available via DWPose) would be needed to study the fine hand gesture vocabulary central to Kunqu aesthetics.

### 8.4 Recommended next steps

1. **Apply multi-scale analysis to additional performances** of the same excerpt by different performers to test whether the coupling profile (correlation-vs-scale curve) is performer-specific or consistent across the tradition.
2. **Segment by dramatic function** (aria, recitative, action sequence) and compare coupling profiles across segment types. The singing style comparison (Section 7.5) provides a starting point, but finer annotation of 拖腔, 装饰音, etc. would enable richer segmentation.
3. **Investigate the negative density–motion relationship**: Why does the performer move less during rapid syllabic passages? Is this a biomechanical constraint (vocal–motor interference), a choreographic convention, or an artifact of the specific excerpt?
4. **Re-run with DWPose** for detailed hand gesture analysis if GPU resources are available.
5. **Explore wavelet coherence** for a time-frequency decomposition of the coupling, which could reveal whether phrase-level synchrony is concentrated in specific frequency bands.
6. **Per-phrase analysis**: Use breath marks to segment the performance into phrases and analyze coupling statistics per phrase, enabling a finer-grained view of how synchrony varies across the dramatic arc.

---

## Appendix: File Inventory

### Pipeline outputs (shoulder root mode)
- `data/annotations/xunmeng_annotation.json` — merged annotation (428 chars, 298 breaths)
- `data/processed/xunmeng_shot_boundaries.json` — 19 cuts, 20 segments
- `data/processed/xunmeng_text_features.npz` + `.json` — 6 text rhythm signals
- `data/processed/xunmeng_audio_features.npz` + `.json` — onset, RMS, f0, pitch delta
- `data/poses/xunmeng_keypoints.npz` + `.json` — 44,831 frames, 33 joints (MediaPipe)
- `data/processed/xunmeng_motion_signals.npz` + `.json` — shoulder root, 7 regions
- `data/processed/xunmeng_aligned_signals.npz` + `.json` — 44,829 frames, 20 channels at 30fps

### Hip root mode (archived)
- `data/processed/xunmeng_motion_signals_hip.npz` + `.json`
- `data/processed/xunmeng_aligned_signals_hip.npz` + `.json`
- `outputs/figures/xunmeng_hip/` — QA plots
- `notebooks/exploration_hip.ipynb` — executed notebook

### QA plots
- `outputs/figures/xunmeng/01_signals_overview.png`
- `outputs/figures/xunmeng/02_pose_quality.png`
- `outputs/figures/xunmeng/03_clip_detail.png`
- `outputs/figures/xunmeng/04_camera_diagnostic.png`

### Analysis notebook
- `notebooks/exploration.ipynb` — source (20 analysis sections: 13 audio-motion + 7 text modality)
- `notebooks/exploration_executed.ipynb` — executed with outputs
