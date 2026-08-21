from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = REPO_ROOT / "downstream" / "report_generation"
LLAVARAD_ROOT = TASK_ROOT / "llavarad"
sys.path.insert(0, str(LLAVARAD_ROOT))

from llava.utils import data_loader_iu_findings, data_loader_mimic_reason_findings
from vision_encoders.utils import VIS_ENCODER_CONFIG, extract_swin_features, get_extract_features_fn


def test_public_csv_loaders(tmp_path, monkeypatch):
    mimic_csv = tmp_path / "mimic.csv"
    mimic_csv.write_text(
        "id,image,view,reason,reference,generate_method\n"
        "1,mimic/p1.png,PA,,clear lungs,rule-based\n"
        "2,p2.png,PA,,,rule-based\n"
    )
    mimic = data_loader_mimic_reason_findings(str(mimic_csv), "test")
    assert len(mimic) == 1
    assert mimic[0]["image"] == "p1.png"
    assert mimic[0]["conversations"][1]["value"] == "clear lungs"

    iu_root = tmp_path / "IUXray"
    labels = iu_root / "labels"
    labels.mkdir(parents=True)
    (iu_root / "indiana_projections.csv").write_text(
        "uid,filename,projection\n10,CXR10.png,Frontal\n"
    )
    iu_csv = labels / "iu.csv"
    iu_csv.write_text(
        "uid,projection,indication,findings\n"
        "10,Frontal,cough,no focal opacity\n"
    )
    monkeypatch.delenv("IU_ROOT", raising=False)
    monkeypatch.delenv("SMARTX_DATA_ROOT", raising=False)
    iu = data_loader_iu_findings(str(iu_csv))
    assert len(iu) == 1
    assert iu[0]["image"] == "CXR10.png"
    assert iu[0]["conversations"][1]["value"] == "no focal opacity"


def test_public_smartx_encoder_alias():
    assert VIS_ENCODER_CONFIG["smartx_large"] == VIS_ENCODER_CONFIG["our_ark_large"]
    assert get_extract_features_fn("smartx_large") is extract_swin_features


def test_public_scripts_match_manuscript_protocol():
    scripts = LLAVARAD_ROOT / "scripts"
    stage1 = (scripts / "train_stage1_custom_mimic.sh").read_text()
    stage2 = (scripts / "train_stage2_custom_mimic.sh").read_text()
    evaluation = (scripts / "eval_custom_mimic.sh").read_text()

    for value in ("--num_train_epochs 1", "--learning_rate 1e-3", "--warmup_ratio 0.03", "--lr_scheduler_type cosine"):
        assert value in stage1
    for value in ("--num_train_epochs 3", "--lora_r 64", "--lora_alpha 128", "--lora_dropout 0.05", "--learning_rate 2e-4", "--model_max_length 2048"):
        assert value in stage2
    for value in ("--temperature 0", "--num_beams 1"):
        assert value in evaluation
    assert "eval_all_checkpoints" not in (scripts / "run_mimic_smartx.sh").read_text()

    merge = (scripts / "merge_llavarad.sh").read_text()
    assert 'export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"' in merge

    for name in ("eval_custom_mimic.sh", "eval_custom_iu.sh"):
        eval_script = (scripts / name).read_text()
        assert 'cat "${prediction_file}"_*.jsonl' not in eval_script
        assert 'cat "${prediction_file}_${idx}.jsonl"' in eval_script

    encoder_source = (LLAVARAD_ROOT / "vision_encoders" / "utils.py").read_text()
    smartx_branch = encoder_source.split(
        'elif model_name.lower() in {"our_ark_large", "smartx_large"}:', 1
    )[1].split("# ResNet50", 1)[0]
    assert "strict=True" in smartx_branch


def test_public_shell_syntax_and_paths():
    scripts = sorted((LLAVARAD_ROOT / "scripts").glob("*.sh"))
    for script in scripts:
        subprocess.run(["bash", "-n", str(script)], check=True)

    tracked_text = "\n".join(
        path.read_text(errors="ignore")
        for path in LLAVARAD_ROOT.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )
    for private_marker in ("/scratch/", "/project/", "/home/chenzhixuan", "zchenhi", "topk_v"):
        assert private_marker not in tracked_text
