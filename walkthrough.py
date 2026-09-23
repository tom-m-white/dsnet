"""Follow ONE video through DSNet end to end, printing what happens at each step.

Run from anywhere:  python walkthrough.py
Uses the authors' pretrained anchor-based checkpoint (tvsum split 0) and a test video.
"""
import sys
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).parent / "DSNet" / "src"
sys.path.insert(0, str(SRC))
from anchor_based import anchor_helper  # noqa: E402
from anchor_based.dsnet import DSNet  # noqa: E402
from helpers import bbox_helper, vsumm_helper  # noqa: E402
import h5py  # noqa: E402

H5 = SRC.parent / "datasets" / "eccv16_dataset_tvsum_google_pool5.h5"
CKPT = SRC.parent / "models" / "pretrain_ab_basic" / "checkpoint" / "tvsum.yml.0.pt"
VIDEO = "video_35"  # in the TEST set of tvsum split 0, so the model never trained on it

np.set_printoptions(precision=2, suppress=True, linewidth=110)

with h5py.File(H5, "r") as f:
    v = f[VIDEO]
    seq = v["features"][...].astype(np.float32)
    gtscore = v["gtscore"][...]
    cps = v["change_points"][...]
    nfps = v["n_frame_per_seg"][...]
    n_frames = int(v["n_frames"][()])
    picks = v["picks"][...]
    user_summary = v["user_summary"][...]

print(f"STEP 0  INPUT  {VIDEO}: {n_frames} original frames -> {len(seq)} sampled steps "
      f"x {seq.shape[1]} features")
print("        first 8 feature values of step 0:", seq[0, :8])
print(f"        video is pre-cut into {len(cps)} shots (KTS), first 3: {cps[:3].tolist()}")

model = DSNet("attention", 1024, 128, [4, 8, 16, 32], 8)
model.load_state_dict(torch.load(CKPT, map_location="cpu"))
model.eval()
x = torch.from_numpy(seq).unsqueeze(0)

with torch.no_grad():
    ctx = model.base_model(x)
    print(f"\nSTEP 1  ATTENTION  every step looks at every other step -> still {tuple(ctx.shape)}")
    print("        (each row is now 'this frame, in the context of the whole video')")

    pred_cls, pred_loc = model(x)
print(f"\nSTEP 2  ANCHORS  at each of {len(seq)} steps, 4 candidate windows of width "
      f"4/8/16/32 steps (= {4 * 15}/{8 * 15}/{16 * 15}/{32 * 15} frames)")
print(f"        -> {pred_cls.numel()} candidate segments. For each the model outputs:")
print("        score (0-1 'is this important?') + 2 offsets (shift centre, stretch width)")
t = int(pred_cls.max(1).values.argmax())
print(f"        e.g. step {t}: scores for widths 4/8/16/32 = {pred_cls[t].numpy()}")

with torch.no_grad():
    pred_cls, pred_bboxes = model.predict(x)
pred_bboxes = np.clip(pred_bboxes, 0, len(seq)).round().astype(np.int32)
best = pred_cls.argmax()
print(f"\nSTEP 3  REFINE  anchor + offsets -> real segment. Best one: steps "
      f"{pred_bboxes[best].tolist()} score {pred_cls[best]:.2f}")

kept_cls, kept_bboxes = bbox_helper.nms(pred_cls, pred_bboxes, 0.5)
print(f"\nSTEP 4  NMS  drop overlapping duplicates: {len(pred_cls)} -> {len(kept_cls)} segments")
for c, b in list(zip(kept_cls, kept_bboxes))[:5]:
    print(f"        steps {b[0]:4d}-{b[1]:4d}  score {c:.2f}")

summ = vsumm_helper.bbox2summary(len(seq), kept_cls, kept_bboxes, cps, n_frames, nfps, picks)
chosen = [i for i, (a, b) in enumerate(cps) if summ[a:b + 1].all()]
print(f"\nSTEP 5  KNAPSACK  segment scores -> shot scores; pick shots whose total length <= 15%")
print(f"        chose {len(chosen)}/{len(cps)} shots = {summ.sum()} frames "
      f"= {100 * summ.mean():.1f}% of the video. Shot ids: {chosen}")

f1s = [vsumm_helper.f1_score(summ, u) for u in user_summary]
print(f"\nSTEP 6  SCORE  compare with each of {len(user_summary)} human summaries (F1 overlap):")
print("        ", np.array(f1s))
print(f"        TVSum uses the AVERAGE -> F-score {np.mean(f1s):.3f}   "
      f"(SumMe would use the MAX -> {np.max(f1s):.3f})")

print("\nFOR COMPARISON  how do humans agree with EACH OTHER? (leave-one-out, same F1)")
human = [np.mean([vsumm_helper.f1_score(u, o) for j, o in enumerate(user_summary) if j != i])
         for i, u in enumerate(user_summary)]
print(f"        average human-vs-other-humans F1: {np.mean(human):.3f}")

print("\nTRAINING TARGET  (what the model learned from, for comparison)")
target = vsumm_helper.downsample_summ(
    vsumm_helper.get_keyshot_summ(gtscore / gtscore.max(), cps, n_frames, nfps, picks))
print(f"        gtscore -> knapsack -> {len(bbox_helper.seq2bbox(target))} target segments, e.g.",
      bbox_helper.seq2bbox(target)[:4].tolist())
