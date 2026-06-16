from __future__ import annotations

TRIAGE_LABEL_TO_ID = {
    "negative": 0,
    "non-emergency-positive": 1,
    "non_emergency_positive": 1,
    "non emergency positive": 1,
    "positive": 2,
}
TRIAGE_ID_TO_LABEL = {0: "negative", 1: "non-emergency-positive", 2: "positive"}

# Historical code used this ordering. It is retained only for checkpoint/data migration.
LEGACY_TRIAGE_LABEL_TO_ID = {
    "negative": 0,
    "positive": 1,
    "non-emergency-positive": 2,
    "non_emergency_positive": 2,
    "non emergency positive": 2,
}

CQ500_LABELS = [
    "ICH",
    "IPH",
    "IVH",
    "SDH",
    "EDH",
    "SAH",
    "BleedLocation-Left",
    "BleedLocation-Right",
    "ChronicBleed",
    "Fracture",
    "CalvarialFracture",
    "OtherFracture",
    "MassEffect",
    "MidlineShift",
]

MANIFEST_ALIASES = {
    "image_path": ["image_path", "Image Path", "path", "volume_path", "ct_path"],
    "report": ["report", "Answer", "answer", "radiology_report", "text"],
    "question": ["question", "Question"],
    "triage_label": ["triage_label", "class_label", "label"],
    "sample_id": ["sample_id", "study_id", "Study ID", "id", "accession", "uid"],
    "split": ["split", "Split"],
}
