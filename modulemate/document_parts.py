"""Static Office text/data that high-level paragraph APIs leave out."""

from __future__ import annotations

from xml.etree import ElementTree as ET

from modulemate.db import UserError

WORD = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
DRAW = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
CHART = "{http://schemas.openxmlformats.org/drawingml/2006/chart}"
REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def paragraph_text(root, namespace=WORD):
    def inline(node):
        if node.tag == namespace + "t":
            return node.text or ""
        if node.tag in {namespace + "br", namespace + "cr"}:
            return "\n"
        if node.tag == namespace + "tab":
            return "\t"
        return "".join(inline(child) for child in node if child.tag != namespace + "p")

    return "\n".join(inline(paragraph) for paragraph in root.iter(namespace + "p")).strip()


def word_additional_parts(document):
    """Yield only referenced headers/notes and body text boxes, never URLs.

    DOCX has no trustworthy page numbering. These locations describe the
    source's actual XML part/identifier instead of inventing page numbers.
    """
    for i, root in enumerate(document.element.iter(WORD + "txbxContent"), 1):
        yield f"text box {i}", paragraph_text(root), root, document.part
    seen = set()
    for kind in ("header", "footer"):
        for element in document.element.iter(WORD + kind + "Reference"):
            rid = element.get(REL + "id")
            relation = document.part.rels.get(rid)
            if relation is None or relation.is_external:
                continue
            part = relation.target_part
            if part.partname in seen:
                continue
            seen.add(part.partname)
            root = ET.fromstring(part.blob)
            variant = element.get(WORD + "type", "default")
            yield f"{kind} {variant} {len(seen)}", paragraph_text(root), root, part
    for kind in ("footnote", "endnote"):
        referenced = {
            node.get(WORD + "id") for node in document.element.iter(WORD + kind + "Reference")
        }
        if not referenced:
            continue
        for relation in document.part.rels.values():
            if relation.is_external or not relation.reltype.endswith("/" + kind + "s"):
                continue
            part = relation.target_part
            root = ET.fromstring(part.blob)
            for note in root.iter(WORD + kind):
                identity = note.get(WORD + "id", "")
                if identity in referenced and identity.isdigit() and int(identity) > 0:
                    yield f"{kind} {identity}", paragraph_text(note), note, part


def embedded_images(root, part):
    for element in root.iter(DRAW + "blip"):
        rid = element.get(REL + "embed")
        image = part.related_parts.get(rid) if rid else None
        if image is not None:
            yield image.blob


def chart_text(chart):
    """Read cached chart values and their labels, without running a workbook.

    Values are explicitly raw: percentage/currency/date formatting can change
    the displayed value. Store format codes alongside their own field.
    """
    root = ET.fromstring(chart._chartSpace.xml)
    lines = ["# Native chart data", "Cached values; display formatting is not applied."]
    labels = [node.text for node in root.iter(DRAW + "t") if node.text]
    if labels:
        lines.append("Chart/axis text: " + " | ".join(labels))
    points = 0
    for index, series in enumerate(root.iter(CHART + "ser"), 1):
        names = series.find(CHART + "tx")
        name = (
            " ".join(node.text or "" for node in names.iter(CHART + "v"))
            if names is not None
            else ""
        )
        lines.append(f"## Series {index}: {name.strip() or 'unnamed'}")
        records = {}
        for field, label in (
            ("cat", "Category"),
            ("val", "Value"),
            ("xVal", "X"),
            ("yVal", "Y"),
            ("bubbleSize", "Bubble size"),
        ):
            element = series.find(CHART + field)
            if element is None:
                continue
            formats = list(
                dict.fromkeys(node.text for node in element.iter(CHART + "formatCode") if node.text)
            )
            if formats:
                lines.append(f"{label} number format: " + " | ".join(formats))
            for point in element.iter(CHART + "pt"):
                value = point.find(CHART + "v")
                identity = point.get("idx", "")
                if value is not None and value.text and identity.isdigit():
                    records.setdefault(int(identity), {}).setdefault(label, []).append(value.text)
                    points += 1
                    if points > 10000:
                        raise UserError(
                            "A chart has more than 10,000 cached points. Split this presentation."
                        )
        for identity, values in sorted(records.items()):
            lines.append(
                f"Point {identity}: "
                + "; ".join(label + " = " + " / ".join(value) for label, value in values.items())
            )
    return "\n".join(lines)


def smartart_text(slide):
    references = dict.fromkeys(
        node.get(REL + "dm")
        for node in slide._element.iter(
            "{http://schemas.openxmlformats.org/drawingml/2006/diagram}relIds"
        )
    )
    for identity in references:
        relation = slide.part.rels.get(identity)
        if relation is None:
            continue
        if not relation.is_external and relation.reltype.endswith("/diagramData"):
            root = ET.fromstring(relation.target_part.blob)
            text = paragraph_text(root, DRAW)
            if text:
                yield text
