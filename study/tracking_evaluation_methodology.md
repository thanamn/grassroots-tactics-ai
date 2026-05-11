# Tracking Evaluation Methodology

This note documents the current PSG vs Milan evaluation set and the expanded
detector comparison runnable in this repo.

## Current Dataset

- Manifest: `data/eval/manifests/psg_annotation_v1.json`
- Labels: `data/eval/annotations/psg_annotation_v1.json`
- Clips: `psg_eval_01`, `psg_eval_02`, `psg_eval_03`
- Clip length: 20 seconds each
- Annotated frames: 40 total
- Ground-truth player points: 840 total
- Player labels per frame: 21 players, each with `id`, `x`, `y`, and `team`
- Ball labels: 39 visible ball labels and 1 frame without a ball label
- Sequence labels: one 4-frame short sequence per clip for continuity scoring

This is enough for a first player-tracking, team-assignment, and ball-locality
benchmark. It is still not enough for true pass count, pass accuracy, turnover
accuracy, or possession-owner accuracy because those need explicit event and
possession labels.

## Player vs Ball Models

The current production software uses the same football-specific detector pass
for player tracking and ball tracking:

- `models/football_players.pt`
- source: `uisikdag/yolo-v8-football-players-detection`
- class `0`: ball
- class `1`: goalkeeper
- class `2`: outfield player

Generic COCO models are different. In YOLO/RT-DETR COCO weights, class `0` is
`person`, not `ball`. Therefore generic models are valid player-detection
baselines, but they are not valid ball-tracking baselines unless they are
fine-tuned on a football-ball class. The ball scorer keeps them in the report
as "no-ball" controls, but the meaningful ball comparison is between the
football-specific confidence settings.

The downloaded Hugging Face checkpoint
`models/football_hf/uisikdag__yolo-v8-football-players-detection__best.pt` has
the same SHA-256 hash as `models/football_players.pt`, so the production model
is treated as the uisikdag model in the fair comparison.

## Fair Football-Finetuned Comparison

Preset file:

- `study/football_finetuned_model_presets.json`

This is the main model-comparison table to use in the paper. It excludes
generic COCO detectors from the main fairness claim and compares only models
with football/soccer-specific labels.

Sources and class maps:

| Preset | Source | Player classes | Ball classes | Notes |
|---|---|---:|---:|---|
| `uisikdag_yolov8_football_players` | https://huggingface.co/uisikdag/yolo-v8-football-players-detection | `[1, 2]` | `[0]` | production source model |
| `tmoklc_football_players` | https://huggingface.co/tmoklc/football-player-detection | `[1, 2]` | `[0]` | multiclass football detector |
| `maerie88_football_players` | https://huggingface.co/maerie88/FOOTBALL-PLAYER-DETECTION-MODEL | `[1, 2]` | `[0]` | compact multiclass detector |
| `martinjolif_yolo11m_football_players` | https://huggingface.co/martinjolif/yolo-football-player-detection | `[1, 2]` | `[0]` | YOLO11m football detector |
| `hamza_yolov8_football_players` | https://huggingface.co/HamzaAliKhan/football-players-detection | `[1, 2]` | `[0]` | multiclass football detector |
| `mobadam_yolo26l_football_players` | https://huggingface.co/mobadam/football-player-detection | `[1, 3]` | `[0]` | YOLO26l; goalkeeper is class `3` |
| `asahwells_yolo26n_football` | https://huggingface.co/asahwells/yolo26n-football-detection | `[1, 2]` | `[0]` | YOLO26n football detector |
| `jhsu12_yolov8n_football` | https://huggingface.co/jhsu12/yolov8n-football-finetuned | `[1, 2]` | `[0]` | high-recall, low-precision detector |
| `hoseinshr_yolov8m_player_only` | https://huggingface.co/hoseinshr1055/yolov8-football-detection | `[0]` | `[]` | player-only, no ball class |
| `parsagh_yolov8_football` | https://huggingface.co/ParsaGh/yolov8_football | `[1]` | `[0]` | ball/player/referee/other labels |
| `martinjolif_yolo11n_ball` | https://huggingface.co/martinjolif/yolo-football-ball-detection | `[]` | `[0]` | ball-only detector |
| `shahzain_soccerball_detector` | https://huggingface.co/ShahzainHaider/football_detection | `[]` | `[1]` | class `1` is soccerball |
| `rajat_soccer_ball` | https://huggingface.co/RajatDave/soccer-ball-detection | `[]` | `[0]` | ball-only detector |

Rejected/secondary candidates:

- `hoseinshr1055/yolov8-football-detection::yolov8_football.pt` loaded as COCO names, so it is not treated as football-finetuned.
- `RaghavRaahul/soccer_ball_detection::checkpoint_5.pt` loaded as COCO names, so it is not treated as a local ball model.
- Roboflow model cards without local weights were not included in the runnable benchmark.
- `lenhat543/DeTR_football_object_detection` is a Hugging Face DETR checkpoint, not an Ultralytics YOLO checkpoint; it needs a separate transformer inference adapter before fair runtime scoring.

Fair run:

- Run: `psg_tracking_football_finetuned_v1`
- Player report: `data/eval/reports/psg_annotation_v1_psg_tracking_football_finetuned_v1_tracking_summary.csv`
- Ball report: `data/eval/reports/psg_annotation_v1_psg_tracking_football_finetuned_v1_ball_summary.csv`
- Baseline: `uisikdag_yolov8_football_players`

Player-capable results:

| Model | Coverage | Precision | F1 | Team acc. | Continuity | Mean err px | Runtime s/clip | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `hamza_yolov8_football_players` | 0.9821 | 0.9167 | 0.9483 | 0.8388 | 0.8939 | 7.766 | 122.265 | 4.098 |
| `martinjolif_yolo11m_football_players` | 0.9940 | 0.8950 | 0.9419 | 0.8395 | 0.9312 | 7.274 | 358.330 | 1.413 |
| `tmoklc_football_players` | 0.9821 | 0.8900 | 0.9338 | 0.8436 | 0.9050 | 7.211 | 1047.230 | 0.479 |
| `maerie88_football_players` | 0.9738 | 0.8920 | 0.9311 | 0.8362 | 0.9389 | 7.724 | 62.710 | 7.992 |
| `hoseinshr_yolov8m_player_only` | 0.9786 | 0.8867 | 0.9304 | 0.8443 | 0.9781 | 7.565 | 609.803 | 0.832 |
| `uisikdag_yolov8_football_players` | 0.9464 | 0.9138 | 0.9298 | 0.8340 | 0.9521 | 7.623 | 76.574 | 6.543 |
| `parsagh_yolov8_football` | 0.9214 | 0.8736 | 0.8969 | 0.8398 | 0.9412 | 7.849 | 58.000 | 8.638 |
| `mobadam_yolo26l_football_players` | 0.8917 | 0.8981 | 0.8949 | 0.8825 | 0.9814 | 7.407 | 547.686 | 0.915 |
| `asahwells_yolo26n_football` | 0.9500 | 0.8444 | 0.8941 | 0.8283 | 0.8528 | 7.040 | 62.415 | 8.027 |
| `jhsu12_yolov8n_football` | 0.9964 | 0.7896 | 0.8811 | 0.8459 | 0.8182 | 7.242 | 1037.514 | 0.483 |

Player interpretation:

- `hamza_yolov8_football_players` has the best F1 on the current PSG sample.
- `uisikdag_yolov8_football_players` remains a strong production default because it has high precision, good F1, and much lower CPU cost than the heavy alternatives.
- `maerie88_football_players` is a strong lightweight challenger: similar F1 to uisikdag, slightly higher coverage, slightly lower precision, and faster runtime.
- `tmoklc`, `jhsu12`, `hoseinshr`, and `mobadam` are too slow for a normal CPU workflow despite interesting recall/continuity behavior.
- The paired bootstrap flags only a few practically meaningful differences on 40 frames, so treat model ranking as engineering evidence rather than final statistical proof.

Ball-capable results:

| Model | Ball recall | Within 50 px | Mean err px | Median err px | Proxy poss. agreement | Full metric clips |
|---|---:|---:|---:|---:|---:|---:|
| `mobadam_yolo26l_football_players` | 0.9744 | 0.7949 | 46.068 | 3.497 | 0.7692 | 3 |
| `martinjolif_yolo11n_ball` | 0.9744 | 0.7179 | 63.962 | 3.471 | 0.0769 | 3 |
| `martinjolif_yolo11m_football_players` | 0.9744 | 0.6923 | 116.953 | 3.795 | 0.6923 | 3 |
| `asahwells_yolo26n_football` | 0.9487 | 0.6667 | 151.136 | 5.697 | 0.6923 | 3 |
| `tmoklc_football_players` | 0.9744 | 0.6667 | 122.884 | 4.751 | 0.6923 | 3 |
| `hamza_yolov8_football_players` | 1.0000 | 0.5897 | 197.951 | 5.487 | 0.7436 | 3 |
| `jhsu12_yolov8n_football` | 0.9231 | 0.5128 | 230.052 | 8.765 | 0.6923 | 3 |
| `uisikdag_yolov8_football_players` | 0.7949 | 0.4615 | 133.715 | 25.577 | 0.5641 | 3 |
| `maerie88_football_players` | 0.6923 | 0.3846 | 118.819 | 25.122 | 0.4872 | 3 |
| `parsagh_yolov8_football` | 0.0000 | 0.0000 | n/a | n/a | 0.0000 | 0 |
| `shahzain_soccerball_detector` | 0.7692 | 0.0000 | 597.164 | 521.043 | 0.0769 | 3 |
| `rajat_soccer_ball` | 0.5385 | 0.0000 | 348.994 | 378.896 | 0.0256 | 3 |

Ball interpretation:

- `mobadam_yolo26l_football_players` is the strongest ball model on these clips, but it is very slow on CPU.
- `martinjolif_yolo11n_ball` is a strong ball-only model. It cannot provide possession by itself because it does not detect players, but it could be paired with a separate player model later.
- `hamza_yolov8_football_players` detects the ball on every visible-ball frame but has more large localization errors, so recall alone is not enough.
- `shahzain_soccerball_detector` and `rajat_soccer_ball` are not suitable for this broadcast PSG sample despite being ball-specific.
- True pass count and possession accuracy still need manual possession/event labels; current possession agreement is a nearest-player proxy only.

## Models Compared

Preset file:

- `study/model_presets_10plus.json`

Completed full-run presets:

- `football_players`: deployed football-specific model, default confidence.
- `football_players_conf015`: same football model at confidence `0.15`, testing recall-heavy behavior.
- `football_players_conf040`: same football model at confidence `0.40`, testing precision-heavy behavior.
- `yolo11n`, `yolo11s`, `yolo11m`, `yolo11l`, `yolo11x`: YOLO11 COCO person baselines from fast to very slow.
- `yolov8n`, `yolov8s`, `yolov8m`: older YOLOv8 COCO person baselines.
- `yolov10n`, `yolov10s`, `yolov10m`: YOLOv10 COCO person baselines.
- `rtdetr_l`: RT-DETR large COCO person baseline, included as a different transformer-style detector architecture.

External model references used for model selection:

- Ultralytics YOLO11 docs: https://docs.ultralytics.com/models/yolo11/
- Ultralytics YOLOv8 docs: https://docs.ultralytics.com/models/yolov8/
- Ultralytics YOLOv10 docs: https://docs.ultralytics.com/models/yolov10/
- Ultralytics RT-DETR docs: https://docs.ultralytics.com/models/rtdetr/

## Commands

Download/verify the extra weights by loading them with Ultralytics from the
repo root:

```powershell
@'
from ultralytics import YOLO
for name in [
    "yolov8n.pt", "yolov8s.pt", "yolov8m.pt",
    "yolov10n.pt", "yolov10s.pt", "yolov10m.pt",
    "yolo11l.pt", "yolo11x.pt", "rtdetr-l.pt",
]:
    print("loading", name)
    YOLO(name)
'@ | & "C:\Users\thana\Desktop\grassroots-tactics-ai\.venv\Scripts\python.exe" -
```

Run the expanded detector comparison:

```powershell
$presets = @(
  "football_players",
  "football_players_conf015",
  "football_players_conf040",
  "yolo11n",
  "yolo11s",
  "yolo11m",
  "yolo11l",
  "yolo11x",
  "yolov8n",
  "yolov8s",
  "yolov8m",
  "yolov10n",
  "yolov10s",
  "yolov10m",
  "rtdetr_l"
)

foreach ($preset in $presets) {
  & "C:\Users\thana\Desktop\grassroots-tactics-ai\.venv\Scripts\python.exe" `
    scripts\run_tracking_benchmark.py `
    --manifest-id psg_annotation_v1 `
    --run-id psg_tracking_latest_v1 `
    --preset-file study\model_presets_10plus.json `
    --preset $preset
}
```

Score the comparison:

```powershell
& "C:\Users\thana\Desktop\grassroots-tactics-ai\.venv\Scripts\python.exe" `
  scripts\score_tracking_benchmark.py `
  --manifest-id psg_annotation_v1 `
  --run-id psg_tracking_latest_v1 `
  --baseline football_players `
  --bootstrap-iterations 5000

& "C:\Users\thana\Desktop\grassroots-tactics-ai\.venv\Scripts\python.exe" `
  scripts\score_ball_benchmark.py `
  --manifest-id psg_annotation_v1 `
  --run-id psg_tracking_latest_v1
```

## Metrics

All player matching uses Hungarian assignment between ground-truth player
points and predicted player points. A match is accepted only when the pixel
distance is at most `35 px`.

- `coverage`: matched ground-truth players divided by all ground-truth players.
- `false_positives`: predicted players that were not matched to any ground-truth player.
- `false_positive_per_gt`: false positives divided by ground-truth players.
- `precision`: matched predictions divided by all scored predictions.
- `f1`: harmonic mean of coverage and precision.
- `team_accuracy`: team-correct matched predictions divided by matched predictions, after per-clip A/B alignment.
- `continuity`: same ground-truth ID maps to the same predicted track ID across consecutive annotated sequence frames.
- `mean_error_px`: mean pixel distance for accepted matches.
- `median_error_px`: median pixel distance for accepted matches.
- `avg_runtime_s`: mean elapsed wall-clock time per clip, including tracking and team assignment.
- `avg_processed_fps`: processed frames per elapsed second.

Team labels need a special rule. The clustering stage may call the two teams
`A` and `B` in either order, so the scorer chooses direct or swapped team
mapping per clip before calculating team accuracy. This evaluates whether the
software grouped players correctly, not whether the arbitrary cluster name
matched the annotation name.

Ball metrics use the manually labeled ball center:

- `ball_recall`: predicted ball exists on frames manually marked with a visible ball.
- `ball_within_50px`: predicted ball exists and is within `50 px` of the manual ball.
- `mean_ball_error_px`: mean pixel error for frames where a ball was predicted.
- `median_ball_error_px`: median pixel error for frames where a ball was predicted.
- `proxy_possession_agreement`: nearest-manual-player possession proxy agreement; useful only as a heuristic.
- `full_metric_clips_available`: whether the software could compute full clip-level ball metrics.

## Comparison Rule

Use `football_players` as the baseline because it is the product configuration
when the football weights are present.

A model difference is treated as meaningful only when both are true:

- the paired bootstrap 95% confidence interval for the frame-level difference
  does not cross `0`;
- the absolute difference passes a practical threshold.

Practical thresholds:

- coverage: `0.05`
- precision: `0.05`
- team accuracy: `0.05`
- mean localization error: `5 px`

Runtime is reported as elapsed wall-clock time, not used as the main
significance test. Runtime becomes a decision factor when two models are within
the practical quality thresholds. A model can be preferred for speed only if it
does not meaningfully reduce coverage or precision.

Important caveat: the current dataset has only 40 annotated frames and short
continuity sequences. The bootstrap gives useful engineering evidence, but the
paper should not oversell it as population-level statistical proof.

## Current Player Results

Latest run:

- Run: `psg_tracking_latest_v1`
- Summary: `data/eval/reports/psg_annotation_v1_psg_tracking_latest_v1_tracking_summary.csv`
- Comparisons: `data/eval/reports/psg_annotation_v1_psg_tracking_latest_v1_tracking_comparisons.csv`

| Model | Coverage | Precision | F1 | Team acc. | Continuity | Mean err px | Runtime s/clip | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `football_players` | 0.9464 | 0.9138 | 0.9298 | 0.8340 | 0.9521 | 7.623 | 58.921 | 8.504 |
| `football_players_conf040` | 0.9298 | 0.9287 | 0.9292 | 0.8348 | 0.9634 | 7.684 | 80.334 | 6.237 |
| `football_players_conf015` | 0.9512 | 0.8988 | 0.9242 | 0.8348 | 0.9419 | 7.639 | 80.542 | 6.221 |
| `rtdetr_l` | 0.9214 | 0.7663 | 0.8368 | 0.8385 | 0.9388 | 7.448 | 317.098 | 1.580 |
| `yolov8m` | 0.7619 | 0.7459 | 0.7538 | 0.8313 | 0.9550 | 7.562 | 118.938 | 4.212 |
| `yolo11l` | 0.7250 | 0.7373 | 0.7311 | 0.8374 | 0.9811 | 7.577 | 157.765 | 3.175 |
| `yolo11x` | 0.7024 | 0.7593 | 0.7297 | 0.8373 | 0.9901 | 7.716 | 295.748 | 1.694 |
| `yolo11s` | 0.7107 | 0.7167 | 0.7137 | 0.8308 | 0.9592 | 7.744 | 55.027 | 9.105 |
| `yolo11m` | 0.6833 | 0.7340 | 0.7078 | 0.8484 | 0.9785 | 7.548 | 101.458 | 5.010 |
| `yolov10m` | 0.5762 | 0.7149 | 0.6381 | 0.8512 | 0.9600 | 7.208 | 100.845 | 5.003 |
| `yolov10s` | 0.5036 | 0.6957 | 0.5843 | 0.8203 | 0.9589 | 7.461 | 53.029 | 9.448 |
| `yolov8s` | 0.4357 | 0.6854 | 0.5328 | 0.8197 | 0.9730 | 7.409 | 55.841 | 8.972 |
| `yolov8n` | 0.3869 | 0.6915 | 0.4962 | 0.7938 | 0.9219 | 7.835 | 29.464 | 17.007 |
| `yolo11n` | 0.2714 | 0.6264 | 0.3787 | 0.8509 | 0.8710 | 8.368 | 28.270 | 17.915 |
| `yolov10n` | 0.2476 | 0.6154 | 0.3531 | 0.7981 | 1.0000 | 7.976 | 29.230 | 17.394 |

Interpretation:

- The default `football_players` model remains the best product setting for spacing analysis because it has the strongest F1 while keeping high coverage and precision.
- `football_players_conf015` slightly increases coverage but loses precision; the difference is not practically meaningful under the comparison rule.
- `football_players_conf040` slightly increases precision but loses coverage; the difference is not practically meaningful under the comparison rule.
- `rtdetr_l` has the best generic coverage and a genuinely different architecture, but it is about 5.4x slower than the football model and has much lower precision.
- Larger generic models do not solve the domain gap. They can localize matched players accurately, but they either miss too many players or add too many false positives.
- The fastest generic models (`yolo11n`, `yolov8n`, `yolov10n`) are useful speed baselines, not viable spacing-analysis defaults on these clips.
- Runtime should not be used alone: `yolo11n` is fastest, but its F1 is less than half the football model's F1.

Significance summary:

- All generic YOLO and RT-DETR models have a meaningful precision drop against `football_players`.
- All generic YOLO models have a meaningful coverage drop against `football_players`.
- `rtdetr_l` does not cross the practical coverage-loss threshold in the current 40-frame sample, but its precision loss is meaningful.
- Neither football confidence variant meaningfully beats the default overall.

## Current Ball Results

Latest ball reports:

- `data/eval/reports/psg_annotation_v1_psg_tracking_latest_v1_ball_summary.csv`
- `data/eval/reports/psg_annotation_v1_psg_tracking_latest_v1_ball_frame_metrics.csv`
- `data/eval/reports/psg_annotation_v1_psg_tracking_latest_v1_ball_clip_metrics.csv`

| Model | Ball recall | Within 50 px | Mean err px | Median err px | Proxy poss. agreement | Full metric clips |
|---|---:|---:|---:|---:|---:|---:|
| `football_players` | 0.7949 | 0.4615 | 133.715 | 25.577 | 0.5641 | 3/3 |
| `football_players_conf015` | 0.8462 | 0.4359 | 187.978 | 30.580 | 0.6410 | 3/3 |
| `football_players_conf040` | 0.6667 | 0.4615 | 89.544 | 21.339 | 0.5128 | 3/3 |
| Generic COCO baselines | 0.0000 | 0.0000 | n/a | n/a | 0.0000 | 0/3 |

Ball interpretation:

- The football-specific model is the only tested model family that produces real ball metrics.
- Lower confidence finds the ball more often but increases large-error ball predictions.
- Higher confidence gives cleaner detected ball locations but misses more visible balls.
- The default football setting is the safest paper baseline until we add explicit possession and pass-event labels.
- Generic COCO YOLO/RT-DETR models should not be presented as failed ball trackers; they simply do not contain a football-ball class.

## Next Annotation Work

To make the evaluation stronger:

- Add 2 to 3 more annotated player sequences per clip, ideally through occlusion or camera motion, for stronger continuity evidence.
- Add explicit possession owner labels on selected frames: `A`, `B`, `contested`, or `absent`.
- Add clip-level manual counts for completed passes and turnovers.
- Add event timestamps for passes, turnovers, and possession changes if pass-count validation is important.
- Add at least 2 grassroots-style clips, because the current set is controlled broadcast footage.

The browser annotator is available at `/annotator` when the FastAPI app is
running. Player labels remain in `frames[frame_id].points`; ball labels are
stored separately in `frames[frame_id].ball`, and possession labels are stored
in `frames[frame_id].possession`. This keeps the existing player scorer
compatible while leaving clean data for ball/pass scoring.
