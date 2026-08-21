# Demo Assets

This directory contains public-domain and synthetic demo materials. The table below records each asset source, licence and preprocessing; no private patient data are included.

## Images (`images/`)

All demo X-rays below are from Wikimedia Commons under public-domain / CC0 terms and
contain **no private patient data**. Each file was converted to 8-bit grayscale PNG and
downscaled so the longest side is ≤ 1024 px (aspect ratio preserved); no other editing.

| File | Label | Licence | Source |
| --- | --- | --- | --- |
| `cxr_normal_01.png` | normal | CC0 | https://commons.wikimedia.org/wiki/File:Normal_posteroanterior_(PA)_chest_radiograph_(X-ray).jpg |
| `cxr_pneumonia_01.png` | pneumonia | CC0 | https://commons.wikimedia.org/wiki/File:X-ray_of_lobar_pneumonia.jpg |
| `cxr_effusion_01.png` | pleural effusion | CC0 | https://commons.wikimedia.org/wiki/File:%D0%93%D0%B5%D0%BC%D0%BE%D1%82%D0%BE%D1%80%D0%B0%D0%BA%D1%81,_%D0%BF%D0%B5%D1%80%D0%B5%D0%BB%D0%BE%D0%BC%D1%8B_%D1%80%D1%91%D0%B1%D0%B5%D1%80.jpg (hemothorax = pleural effusion, with rib fractures) |
| `cxr_pneumonia_02.png` | pneumonia | CC0 | https://commons.wikimedia.org/wiki/File:Chest_radiograph_in_influensa_and_H_influenzae,_posteroanterior.jpg |

## Labels / pairs

- `zero_shot_labels.json` — class names with positive/negative prompt templates used by the
  zero-shot demo.
- `retrieval_pairs.jsonl` — one `{image, text}` pair per demo image; the `text` is a short
  synthetic radiology-style report written for the demo (not from any clinical dataset).
