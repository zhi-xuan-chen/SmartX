import datetime
import logging
import logging.handlers
import os
import sys
import json
import csv

import requests

from llava.constants import LOGDIR

server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
moderation_msg = "YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES. PLEASE TRY AGAIN."

handler = None


def build_logger(logger_name, logger_filename):
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format of root handlers
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Add a file handler for all loggers
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(
            filename, when='D', utc=True)
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ''
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == '\n':
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != '':
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ''


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def violates_moderation(text):
    """
    Check whether the text violates OpenAI moderation API.
    """
    url = "https://api.openai.com/v1/moderations"
    headers = {"Content-Type": "application/json",
               "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]}
    text = text.replace("\n", "")
    data = "{" + '"input": ' + f'"{text}"' + "}"
    data = data.encode("utf-8")
    try:
        ret = requests.post(url, headers=headers, data=data, timeout=5)
        flagged = ret.json()["results"][0]["flagged"]
    except requests.exceptions.RequestException as e:
        flagged = False
    except KeyError as e:
        flagged = False

    return flagged


def pretty_print_semaphore(semaphore):
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"


def data_loader_default(data_path):
    logging.info("using the default loader.")
    dataset = json.load(open(data_path, "r"))
    logging.info(f"loaded {len(dataset)} samples.")
    return dataset


def data_loader_mimic_cxr_all_frontal_findings(data_path):
    logging.info("using the MIMIC-CXR loader: all frontal findings.")
    with open(data_path) as f:
        dataset = json.load(f)
    ret = []
    for d in dataset:
        # Skip empty findings
        if not isinstance(d["conversations"][1]["value"], str):
            continue
        if d['view'] in ('AP', 'PA'):
            ret.append(d)
    logging.info(f"loaded {len(ret)}/{len(dataset)} samples.")
    return ret


def data_loader_mimic_cxr_all_views_findings(data_path):
    logging.info("using the MIMIC-CXR loader: all views findings.")
    with open(data_path) as f:
        dataset = json.load(f)
    ret = []
    for d in dataset:
        # Skip empty findings
        if not isinstance(d["conversations"][1]["value"], str):
            continue
        view = d["view"] if isinstance(d["view"], str) else "Unknown"
        d["conversations"][0]["value"] = f"<image>\nGiven the chest X-ray image from {view} view, describe the findings in the image: "
        ret.append(d)
    logging.info(f"loaded {len(ret)}/{len(dataset)} samples.")
    return ret


def data_loader_mimic_reason_findings(data_path, split):
    logging.info(f"using the MIMIC-CXR loader: MIMIC {split}.")
    if data_path.lower().endswith(".csv"):
        with open(data_path, newline="", encoding="utf-8") as f:
            raw_rows = list(csv.DictReader(f))

        dataset = []
        for idx, row in enumerate(raw_rows):
            # CSV schema (id, query, image, reference, ...) -> internal conversation format.
            sample_id = row.get("id", idx)
            image = row.get("image") or row.get("image_path") or ""
            prompt = row.get("query") or "<image>\nProvide a description of the findings in the radiology image."
            reference = row.get("reference") or row.get("findings") or ""
            if not isinstance(reference, str):
                reference = str(reference)
            if isinstance(image, str) and image.startswith("mimic/"):
                image = image[len("mimic/"):]
            if isinstance(image, str) and "/files/" in image:
                image = image.split("/files/", 1)[1]
            dataset.append(
                {
                    "id": sample_id,
                    "image": image,
                    "view": row.get("view", "PA"),
                    "generate_method": row.get("generate_method", "rule-based"),
                    "reason": row.get("reason"),
                    "conversations": [
                        {"from": "human", "value": prompt},
                        {"from": "gpt", "value": reference},
                    ],
                }
            )
    else:
        with open(data_path) as f:
            dataset = json.load(f)
    ret = []
    for d in dataset:
        if split == 'test' and d['generate_method'] != 'rule-based':
            continue
        # Skip empty findings
        if not isinstance(d["conversations"][1]["value"], str):
            continue
        if d['view'] not in ('AP', 'PA'):
            continue
        if d['image'].startswith("mimic/"):
            d['image'] = d['image'][len('mimic/'):]
        if d['reason'] is not None:
            reason = d['reason'].replace('\n', ' ')
            d['conversations'][0]['value'] = f"<image>\nProvide a description of the findings in the radiology image given the following indication: {reason}"
        else:
            d['conversations'][0]['value'] = f"<image>\nProvide a description of the findings in the radiology image."
        ret.append(d)
    logging.info(f"loaded {len(ret)}/{len(dataset)} samples.")
    return ret


def _load_iu_image_map():
    """Build {uid: filename} from indiana_projections.csv (source of truth for filenames)."""
    import os
    iu_root = "${SMARTX_DATA_ROOT}/IUXray"
    img_map = {}  # uid -> filename (prefer Frontal)
    try:
        with open(os.path.join(iu_root, "indiana_projections.csv"), newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                uid = row.get("uid", "")
                fn = row.get("filename", "")
                proj = row.get("projection", "")
                if not fn:
                    continue
                if uid not in img_map:
                    img_map[uid] = fn
                elif proj == "Frontal":
                    img_map[uid] = fn  # Frontal preferred
    except FileNotFoundError:
        pass
    return img_map


def data_loader_iu_findings(data_path):
    """Loader for IU X-Ray CSV data (indiana_meta_frontal_findings_w_chexbert_labels_*.csv).

    Uses indiana_projections.csv as the source of truth for filename mapping.
    """
    logging.info("using the IU X-Ray loader.")
    img_map = _load_iu_image_map()

    with open(data_path, newline="", encoding="utf-8") as f:
        raw_rows = list(csv.DictReader(f))

    dataset = []
    for idx, row in enumerate(raw_rows):
        uid = row.get("uid", idx)
        # Use projections file mapping; fallback to deriving from CSV filename
        image = img_map.get(uid, "")
        if not image:
            filename = row.get("filename", "")
            if not filename:
                continue
            image = filename
            if image.startswith("CXR"):
                image = image[3:]
            image = image.replace(".png", ".dcm.png")

        findings = row.get("findings", "")
        if not isinstance(findings, str):
            findings = str(findings)
        findings = findings.strip().strip('"').strip()
        indication = row.get("indication", "").strip()
        projection = row.get("projection", "Frontal")

        dataset.append({
            "id": uid,
            "image": image,
            "projection": projection,
            "findings": findings,
            "indication": indication,
        })

    ret = []
    for d in dataset:
        if not d["findings"]:
            continue
        if d["projection"] not in ("Frontal", "AP", "PA"):
            continue
        reason = d.get("indication", "")
        if reason:
            d["conversations"] = [
                {"from": "human", "value": f"<image>\nProvide a description of the findings in the radiology image given the following indication: {reason}"},
                {"from": "gpt", "value": d["findings"]},
            ]
        else:
            d["conversations"] = [
                {"from": "human", "value": "<image>\nProvide a description of the findings in the radiology image."},
                {"from": "gpt", "value": d["findings"]},
            ]
        ret.append(d)

    logging.info(f"loaded {len(ret)}/{len(dataset)} samples.")
    return ret


def data_loader_chexpert_findings(data_path, split):
    """Loader for CheXpert-Plus grouped CSV (chexpert_processed_grouped.csv).

    Mirrors the pre-training data convention (pretrain/datasets/CheXpert.py):
    rows are filtered by the `split` column (train / validate); the report
    target is findings_section, falling back to impression_section when
    findings is empty (so rows are kept rather than dropped). The frontal view
    is picked as the single input image. The returned `image` is relative to
    --image_folder (e.g. CheXpert-Plus/imgs-1024), matching MIMIC/IU loaders.
    """
    import ast
    logging.info(f"using the CheXpert-Plus loader: split={split}.")

    # report/findings fields contain embedded newlines -> large CSV cells.
    csv.field_size_limit(10 ** 9)

    def valid_text(v):
        if v is None:
            return False
        return str(v).strip().lower() not in ("", "nan", "none", "null", "n/a", "-")

    def parse_paths(cell):
        if not cell:
            return []
        s = str(cell).strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                return list(ast.literal_eval(s))
            except Exception:
                try:
                    return json.loads(s.replace("'", '"'))
                except Exception:
                    return []
        return [s]

    def pick_frontal(paths):
        for p in paths:
            if "frontal" in str(p).lower():
                return str(p).strip().lstrip("/")
        return str(paths[0]).strip().lstrip("/") if paths else ""

    with open(data_path, newline="", encoding="utf-8") as f:
        raw_rows = [row for row in csv.DictReader(f) if (row.get("split") or "") == split]

    ret = []
    for idx, row in enumerate(raw_rows):
        # findings preferred; fall back to impression when findings is empty.
        if valid_text(row.get("findings_section")):
            report, section = str(row["findings_section"]).strip(), "findings"
        elif valid_text(row.get("impression_section")):
            report, section = str(row["impression_section"]).strip(), "impression"
        else:
            continue
        image = pick_frontal(parse_paths(row.get("image_paths", "")))
        if not image or "frontal" not in image.lower():
            continue
        sample_id = row.get("id") or row.get("StudyId") or idx
        ret.append({
            "id": sample_id,
            "image": image,
            "section": section,
            "conversations": [
                {"from": "human", "value": "<image>\nProvide a description of the findings in the radiology image."},
                {"from": "gpt", "value": report},
            ],
        })

    logging.info(f"loaded {len(ret)}/{len(raw_rows)} samples.")
    return ret


data_loaders = {
    "default": data_loader_default,
    "mimic_train_findings": lambda x: data_loader_mimic_reason_findings(x, "train"),
    "mimic_test_findings": lambda x: data_loader_mimic_reason_findings(x, "test"),
    "mimic_cxr_all_frontal_findings": data_loader_mimic_cxr_all_frontal_findings,
    "mimic_cxr_all_views_findings": data_loader_mimic_cxr_all_views_findings,
    "iu_train_findings": data_loader_iu_findings,
    "iu_test_findings": data_loader_iu_findings,
    "chexpert_train_findings": lambda x: data_loader_chexpert_findings(x, "train"),
    "chexpert_test_findings": lambda x: data_loader_chexpert_findings(x, "validate"),
}