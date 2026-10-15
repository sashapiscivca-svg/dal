"""Clip formats, benchmarks and the command line, on small synthetic clips (no datasets needed)."""
import json

import cv2
import numpy as np
import pytest
from test_classic_mosse import scene, texture

from dal import bench, make_tracker
from dal.cli import main
from dal.clips import Clip, load_clip, load_set, read_boxes_xywh, write_boxes_xywh
from dal.image import encode_frames


def _clip(n=40, start=(60, 70), vel=(1.5, 0.5), size=14, seed=0):
    bg = texture(seed)
    fr, boxes = [], []
    for t in range(n):
        cx, cy = start[0] + round(vel[0] * t), start[1] + round(vel[1] * t)
        fr.append(scene(bg, cx, cy, size).clip(0, 255).astype(np.uint8))
        boxes.append((cx, cy, size, size))
    return np.stack(fr), np.array(boxes, float)


def _write_set(root, n_npz=2, with_dir=True, with_video=False):
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n_npz):
        fr, bx = _clip(seed=i)
        np.savez_compressed(root / f"{i + 1}.npz", frames=fr, boxes=bx.astype(np.float32), group=np.array(f"seq{i % 2}"), fps=np.float32(25))
    if with_dir:
        fr, bx = _clip(seed=3)
        d = root / "clip_dir"
        d.mkdir()
        for t, f in enumerate(fr):
            cv2.imwrite(str(d / f"{t:04d}.png"), f)
        write_boxes_xywh(d / "groundtruth.txt", bx)
        (d / "meta.json").write_text(json.dumps({"group": "seqdir", "fps": 30}))
    if with_video:
        fr, bx = _clip(seed=7)
        path = root / "flight.avi"
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (fr.shape[2], fr.shape[1]))
        for f in fr:
            vw.write(cv2.cvtColor(f, cv2.COLOR_GRAY2BGR))
        vw.release()
        write_boxes_xywh(path.with_suffix(".txt"), bx)
    return root


def test_boxes_roundtrip(tmp_path):
    b = np.array([[10.0, 20.0, 6.0, 8.0], [11.5, 21.0, 6.0, 8.0]])
    write_boxes_xywh(tmp_path / "g.txt", b)
    assert np.allclose(read_boxes_xywh(tmp_path / "g.txt"), b, atol=1e-2)
    (tmp_path / "h.txt").write_text("7 16 6 8\n# comment\n8.5\t17\t6\t8\n")
    assert np.allclose(read_boxes_xywh(tmp_path / "h.txt"), b, atol=1e-2)


def test_load_set_formats_order_and_subsets(tmp_path):
    root = _write_set(tmp_path / "set", n_npz=3, with_video=True)
    clips = load_set(root)
    assert [c.name for c in clips] == ["1", "2", "3", "clip_dir", "flight"]
    assert clips[0].group == "seq0" and clips[1].group == "seq1" and clips[3].group == "seqdir" and clips[0].fps == 25
    for c in clips:
        fr = list(c.frames())
        assert len(fr) == len(c) == 40 and fr[0].dtype == np.uint8
    assert [c.name for c in load_set(root, every=2, offset=1)] == ["2", "clip_dir"]
    assert len(load_set(root, limit=2)) == 2


def test_rescale_to_target_size(tmp_path):
    fr, bx = _clip()
    np.savez(tmp_path / "c.npz", frames=fr, boxes=bx)
    c = load_clip(tmp_path / "c.npz").rescaled(7)
    assert c.scale == pytest.approx(0.5) and np.allclose(c.boxes, bx * 0.5)
    f0 = next(c.frames())
    assert f0.shape == (fr.shape[1] // 2, fr.shape[2] // 2)
    assert load_clip(tmp_path / "c.npz").rescaled(100).scale == 1.0       # never upscaled


def test_encoded_npz_clip_streams(tmp_path):
    fr, bx = _clip(n=6)
    blob, off = encode_frames(fr)
    np.savez(tmp_path / "e.npz", enc=blob, enc_off=off, boxes=bx)
    assert (np.stack(list(load_clip(tmp_path / "e.npz").frames())) == fr).all()


def test_short_and_survival_benchmarks_on_easy_clips(tmp_path):
    root = _write_set(tmp_path / "set", n_npz=3)
    clips = load_set(root)
    fac = bench.make_factory("classic:v1", win=32, lr=0.125)
    runs = [bench.run_clip(fac, c) for c in clips]
    s = bench.short_summary(runs)
    assert s["frames"] == 4 * 39 and s["prf"]["center"]["f1"] > 0.95 and s["lost_rate"] < 0.1
    sv = bench.survival_summary(bench.tracks_from_runs(runs), ns=(10, 30), reps=50)
    assert sv["survival"]["30"]["value"] == 1.0 and sv["sequences"] == 3 and sv["time_to_first_loss"]["no_loss_frac"] == 1.0
    assert "Survival" in bench.format_survival("x", sv) and "F1" in bench.format_short("x", s)


def test_gt_corrections_are_reproducible_and_applied(tmp_path):
    fr, bx = _clip()
    a = bench.gt_corrections(bx, 10, "clip-7")
    b = bench.gt_corrections(bx, 10, "clip-7")
    assert sorted(a) == [10, 20, 30] and all(np.allclose(a[k], b[k]) for k in a)
    np.savez(tmp_path / "c.npz", frames=fr, boxes=bx)
    r = bench.run_clip(bench.make_factory("classic:visible", win=32), load_clip(tmp_path / "c.npz"), correct_every=10)
    assert not r["lost"][9] and r["pred"].shape == (39, 4)


def test_make_factory_rejects_unknown():
    with pytest.raises(ValueError):
        bench.make_factory("classic:nope")
    with pytest.raises(ValueError):
        bench.make_factory("magic:x")
    with pytest.raises(ValueError):
        make_tracker("nope")


def test_cli_track_writes_json_and_video(tmp_path, capsys):
    fr, bx = _clip(n=20)
    d = tmp_path / "frames"
    d.mkdir()
    for t, f in enumerate(fr):
        cv2.imwrite(str(d / f"{t}.png"), f)
    x, y, w, h = bx[0, 0] - 7, bx[0, 1] - 7, 14, 14
    rc = main(["track", str(d), "--init", f"{x},{y},{w},{h}", "--json", str(tmp_path / "t.json"), "--out", str(tmp_path / "t.mp4"),
               "--kw", '{"win": 32, "lr": 0.125}'])
    assert rc == 0
    out = json.loads((tmp_path / "t.json").read_text())
    assert out["tracker"] == "classic:thermal" and len(out["frames"]) == 20       # gray input -> thermal profile
    last = out["frames"][-1]["box_xywh"]
    assert abs(last[0] + last[2] / 2 - bx[-1, 0]) < 3 and abs(last[1] + last[3] / 2 - bx[-1, 1]) < 3
    assert (tmp_path / "t.mp4").stat().st_size > 1000
    assert "classic:thermal" in capsys.readouterr().out


def test_cli_bench_and_info(tmp_path, capsys):
    root = _write_set(tmp_path / "set", n_npz=2, with_dir=False)
    assert main(["bench", "short", str(root), "--tracker", "classic:v1", "--kw", '{"win": 32}', "--json", str(tmp_path / "s.json")]) == 0
    res = json.loads((tmp_path / "s.json").read_text())["results"]["classic:v1"]
    assert res["clips"] == 2 and res["prf"]["center"]["f1"] > 0.9
    assert main(["bench", "survival", str(root), "--tracker", "classic:visible", "--reps", "20"]) == 0
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    assert "Survival" in out and "visible" in out


def test_cli_rejects_bad_box():
    with pytest.raises(SystemExit):
        main(["track", "x.mp4", "--init", "1,2,3"])


def test_clip_object_basics():
    c = Clip("nowhere.npz", np.zeros((3, 4)) + [5, 5, 2, 2], "c")
    assert len(c) == 3 and c.group == "c" and c.rescaled(None) is c


def test_video_piece_clip_spec(tmp_path):
    fr, bx = _clip(n=30)
    vid = tmp_path / "videos" / "v.avi"
    vid.parent.mkdir()
    vw = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"MJPG"), 30, (fr.shape[2], fr.shape[1]))
    for f in fr:
        vw.write(cv2.cvtColor(f, cv2.COLOR_GRAY2BGR))
    vw.release()
    spec = {"video": "../videos/v.avi", "start": 5, "count": 20, "scale": 0.5, "color": False, "group": "v", "fps": 30.0,
            "boxes": (bx[5:25] * 0.5).tolist(), "others": [[] for _ in range(20)]}
    d = tmp_path / "set"
    d.mkdir()
    (d / "v_1.clip.json").write_text(json.dumps(spec))
    (c,) = load_set(d)
    frames = list(c.frames())
    assert c.name == "v_1" and len(frames) == 20 and frames[0].shape == (fr.shape[1] // 2, fr.shape[2] // 2) and frames[0].ndim == 2
    assert np.allclose(c.boxes[0], bx[5] * 0.5) and c.others[0].shape == (0, 4)
    r = bench.run_clip(bench.make_factory("classic:thermal", win=32, lr=0.125), c)
    assert r["pred"].shape == (19, 4)


def test_wisard_linking_and_mobdrone_segments():
    pytest.importorskip("scipy")
    from dal.datasets.mobdrone import parse_tracks, pick, segments
    from dal.datasets.wisard import frame_index, is_ir_sequence, link_tracks, site_of
    a = [np.array([[10.0, 10, 6, 8], [50, 50, 6, 8]]), np.array([[51.0, 50, 6, 8], [11, 10, 6, 8]]), np.array([[12.0, 10, 6, 8]])]
    tr = sorted(link_tracks(a), key=lambda t: t["boxes"][0, 0])
    assert len(tr) == 2 and len(tr[0]["boxes"]) == 3 and len(tr[1]["boxes"]) == 2
    assert site_of("200929_Karen_FLIR_IR_3") == "Karen" and is_ir_sequence("200929_Karen_FLIR_IR_3") and frame_index("x_00000036.jpeg") == 36
    coco = {"images": [{"id": i, "file_name": f"DJI_0804_0001_30m_1_{i:06d}.PNG"} for i in range(40)], "annotations": []}
    for i in range(40):
        if i != 20:                                                        # track 0 has a gap at frame 20
            coco["annotations"].append({"image_id": i, "category_id": 1, "bbox": [100 + i, 50, 10, 12], "attributes": {"track_id": 0}})
        coco["annotations"].append({"image_id": i, "category_id": 1, "bbox": [5 + 2 * i, 60, 10, 12], "attributes": {"track_id": 1}})
        coco["annotations"].append({"image_id": i, "category_id": 9, "bbox": [0, 0, 5, 5], "attributes": {"track_id": 7}})   # not a person
    v = parse_tracks(coco)
    assert list(v) == ["DJI_0804_0001_30m_1"] and set(v["DJI_0804_0001_30m_1"]) == {0, 1}
    t = v["DJI_0804_0001_30m_1"]
    assert [(s, len(b)) for s, b in segments(t[0], 640, 480, min_len=5, max_len=100)] == [(0, 20), (21, 19)]
    assert len(segments(t[1], 60, 480, min_len=5, max_len=100)[0][1]) < 40          # leaving the frame ends the piece
    got = pick(t, 640, 480, per_video=1, min_len=5, max_len=15)
    assert len(got) == 1 and len(got[0][2]) == 15
