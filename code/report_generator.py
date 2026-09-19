"""Generate a waste-management report from YOLO detection results.

Reports are built locally from detection counts and class-specific handling
notes. No paid language-model API is required.
"""

from collections import Counter
from datetime import datetime
from typing import Iterable, Mapping, Sequence

CLASS_GUIDANCE = {
    "bottle": {
        "label": "Bottle",
        "stream": "Recycling, after emptying and a quick rinse",
        "handling": (
            "Empty leftover liquid. Recycle plastic bottles with the plastic "
            "stream and glass bottles with glass. Leave caps on only if your "
            "local program accepts them."
        ),
        "safety": "Check for broken glass. Do not crush glass bottles by hand.",
    },
    "carton": {
        "label": "Carton",
        "stream": "Carton or mixed-paper recycling, if accepted locally",
        "handling": (
            "Empty and flatten. Many cartons have a plastic or foil lining, so "
            "confirm whether your city accepts cartons in recycling."
        ),
        "safety": "Keep wet or food-soiled cartons out of clean paper recycling.",
    },
    "dixie_cup": {
        "label": "Paper cup",
        "stream": "Usually landfill or compost, not standard paper recycling",
        "handling": (
            "Most disposable cups have a plastic lining and are not recycled "
            "with mixed paper. Empty them, then landfill unless they are "
            "certified compostable and a compost program is available."
        ),
        "safety": "Watch for hot-liquid residue and lids or straws that must be sorted separately.",
    },
    "paper": {
        "label": "Paper",
        "stream": "Paper recycling when clean and dry",
        "handling": (
            "Stack clean office paper, newspaper, and similar fiber. Remove "
            "tape, staples, and plastic windows when practical. Food-soiled "
            "paper belongs in compost or landfill, not recycling."
        ),
        "safety": "Do not recycle paper that is wet, greasy, or mixed with trash.",
    },
}

GENERIC_GUIDANCE = {
    "label": None,
    "stream": "Follow local sorting rules for this material",
    "handling": (
        "Identify the material, empty any contents, and place it in the "
        "matching recycling, compost, or landfill stream used by the local authority."
    ),
    "safety": "Wear gloves if the item is sharp, dirty, or unknown.",
}


def count_detected_classes(detections: Iterable) -> dict[str, int]:
    """Return {class_name: count} from web.py detInfo rows or model dictionaries."""
    counts: Counter[str] = Counter()
    for detection in detections:
        if isinstance(detection, Mapping):
            name = detection.get("class_name") or detection.get("name")
        elif isinstance(detection, Sequence) and not isinstance(detection, (str, bytes)):
            name = detection[0] if detection else None
        else:
            name = None

        if name and str(name).strip() and str(name).strip().lower() != "none":
            counts[str(name).strip()] += 1

    return dict(sorted(counts.items()))


def build_report_input(class_counts: Mapping[str, int]) -> list[str]:
    """Create inventory lines such as ['bottle: 3', 'paper: 2']."""
    return [f"{name}: {int(count)}" for name, count in class_counts.items() if count > 0]


def _parse_report_input(report_input: Sequence[str]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for item in report_input:
        if ":" not in str(item):
            continue
        name, raw_count = str(item).rsplit(":", 1)
        name = name.strip()
        try:
            count = int(str(raw_count).strip())
        except ValueError:
            continue
        if name and name.lower() != "none" and count > 0:
            counts[name] += count
    return dict(sorted(counts.items()))


def _guidance_for(class_name: str) -> dict[str, str]:
    key = class_name.strip().lower().replace(" ", "_")
    info = CLASS_GUIDANCE.get(key, GENERIC_GUIDANCE).copy()
    info["label"] = info.get("label") or class_name.replace("_", " ").title()
    return info


def generate_waste_report(
    report_input: Sequence[str],
    user_command: str = "",
    class_counts: Mapping[str, int] | None = None,
    **_unused,
) -> str:
    """Build a Markdown waste report from detection counts. Runs fully offline."""
    counts = {
        str(name): int(count)
        for name, count in (class_counts or _parse_report_input(report_input)).items()
        if int(count) > 0
    }
    if not counts:
        raise ValueError("No detected waste classes are available for the report.")

    total = sum(counts.values())
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# Waste Segmentation Report",
        "",
        f"Generated: {generated_at}",
        f"Total detected items: **{total}**",
        "",
        "Counts come from the segmentation model. They are object detections, "
        "not measured weight or volume. Disposal rules vary by local authority.",
        "",
        "## 1. Overview",
        "",
        "The current inspection found the following waste items:",
        "",
    ]
    for name, count in counts.items():
        label = _guidance_for(name)["label"]
        pct = 100.0 * count / total
        lines.append(f"- {label}: {count} ({pct:.1f}%)")

    lines.extend(
        [
            "",
            "## 2. Waste composition",
            "",
            "| Class | Count | Share | Preferred stream |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for name, count in counts.items():
        info = _guidance_for(name)
        pct = 100.0 * count / total
        lines.append(
            f"| {info['label']} | {count} | {pct:.1f}% | {info['stream']} |"
        )

    lines.extend(["", "## 3. Handling and disposal", ""])
    for name, count in counts.items():
        info = _guidance_for(name)
        lines.extend(
            [
                f"### {info['label']} ({count})",
                "",
                f"**Preferred stream:** {info['stream']}",
                "",
                info["handling"],
                "",
            ]
        )

    lines.extend(["## 4. Safety and contamination", ""])
    for name, count in counts.items():
        info = _guidance_for(name)
        lines.append(f"- **{info['label']}:** {info['safety']}")

    lines.extend(
        [
            "",
            "## 5. Operational actions",
            "",
            "1. Separate items by class before placing them in collection bins.",
            "2. Empty liquids and food residue so recyclables stay clean.",
            "3. Photograph or export the detection log for traceability.",
            "4. Confirm the final stream with the local waste authority.",
            "",
        ]
    )

    notes = (user_command or "").strip()
    default_note = (
        "Generate a detailed waste-management report with an overview, "
        "composition analysis, disposal methods, safety precautions, and "
        "actionable operational recommendations."
    )
    if notes and notes != default_note:
        lines.extend(["## Operator notes", "", notes, ""])

    return "\n".join(lines).rstrip() + "\n"
