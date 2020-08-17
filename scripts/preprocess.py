# -*- coding: utf-8 -*-
import json
import os
import string
import time
from collections import OrderedDict, defaultdict
from glob import glob
from html import unescape
import argparse

import cchardet
import regex
from bs4 import BeautifulSoup

import ssplit
from tokenization_bert import BasicTokenizer

os.environ["PYTHONHASHSEED"] = "42"

# EXCEPTIONAL_ENTITY_TYPES = {"Protein_domain_or_region", "DNA_domain_or_region"}

BASIC_TOKENIZER = BasicTokenizer(do_lower_case=False)


def generate_sentence_boundaries(doc):
    offsets = []
    for start_offset, end_offset in ssplit.regex_sentence_boundary_gen(doc):
        # Skip empty lines
        if doc[start_offset:end_offset].strip():

            while doc[start_offset] == " ":
                start_offset += 1

            while doc[end_offset - 1] == " ":
                end_offset -= 1

            assert start_offset < end_offset

            offsets.append((start_offset, end_offset))

    return offsets


def norm_path(*paths):
    return os.path.relpath(os.path.normpath(os.path.join(os.getcwd(), *paths)))


def make_dirs(*paths):
    os.makedirs(norm_path(*paths), exist_ok=True)


def read_file(filename):
    with open(norm_path(filename), "rb") as f:
        return f.read()


def detect_file_encoding(filename):
    return cchardet.detect(read_file(filename))["encoding"]


def read_text(filename, encoding=None):
    encoding = encoding or detect_file_encoding(filename)
    with open(norm_path(filename), "r", encoding=encoding) as f:
        return f.read()


def write_text(text, filename, encoding="UTF-8"):
    make_dirs(os.path.dirname(filename))
    with open(norm_path(filename), "w", encoding=encoding) as f:
        f.write(text)


def read_lines(filename, encoding=None):
    encoding = encoding or detect_file_encoding(filename)
    with open(norm_path(filename), "r", encoding=encoding) as f:
        for line in f:
            yield line.rstrip("\r\n\v")


def write_lines(lines, filename, linesep="\n", encoding="UTF-8"):
    make_dirs(os.path.dirname(filename))
    with open(norm_path(filename), "w", encoding=encoding) as f:
        for line in lines:
            f.write(line)
            f.write(linesep)


def read_json(filename, encoding=None):
    return json.loads(read_text(filename, encoding=encoding))


def write_json(obj, filename, indent=2, encoding="UTF-8"):
    write_text(
        json.dumps(obj, indent=indent, ensure_ascii=False), filename, encoding=encoding
    )


def parse_standoff_file(standoff_file, text_file, encoding=None):
    assert os.path.exists(standoff_file), "Standoff file not found: " + standoff_file
    assert os.path.exists(text_file), "Text file not found: " + text_file

    entities = OrderedDict()
    relations = OrderedDict()
    events = OrderedDict()
    modalities = OrderedDict()
    attributes = OrderedDict()
    equivalences = []

    # Using reference for double-check
    reference = read_text(text_file, encoding=encoding)

    for line in read_lines(standoff_file, encoding=encoding):
        # Trim trailing whitespaces
        line = line.strip()

        if line.startswith(
                "T"
        ):  # Entities (T), Triggers (TR) (are also included in this case)
            entity_id, entity_annotation, entity_reference = line.split("\t")

            entity_id = entity_id.strip()
            entity_annotation = entity_annotation.strip()
            entity_reference = entity_reference.strip()

            annotation_elements = entity_annotation.split(";")
            entity_type, *first_offset_pair = annotation_elements[0].split()

            offset_pairs = [first_offset_pair] + [
                offset_pair.split() for offset_pair in annotation_elements[1:]
            ]

            if len(offset_pairs) > 1:
                print(
                    "## Discontinuous entity found (will be merged to be compatible): {} in {}".format(
                        entity_id, standoff_file
                    )
                )

            start_offsets, end_offsets = list(
                zip(
                    *[
                        (int(start_offset), int(end_offset))
                        for start_offset, end_offset in offset_pairs
                    ]
                )
            )

            start_offset, end_offset = min(start_offsets), max(end_offsets)

            actual_reference = reference[start_offset:end_offset]

            entities[entity_id] = {
                "id": entity_id,
                "type": entity_type,
                "start": start_offset,
                "end": end_offset,
                "ref": actual_reference,
            }
        elif line.startswith("R"):  # Relations
            relation_id, relation_annotation = line.split("\t")

            relation_id = relation_id.strip()
            relation_annotation = relation_annotation.strip()

            relation_role, left_arg, right_arg = relation_annotation.split()

            left_arg_label, left_arg_id = left_arg.split(":")
            right_arg_label, right_arg_id = right_arg.split(":")

            relations[relation_id] = {
                "id": relation_id,
                "role": relation_role,
                "left_arg": {"label": left_arg_label, "id": left_arg_id},
                "right_arg": {"label": right_arg_label, "id": right_arg_id},
            }
        elif line.startswith("E"):  # Events
            event_id, event_annotation = line.split("\t")

            event_id = event_id.strip()
            event_annotation = event_annotation.strip()

            trigger, *args = event_annotation.split()

            trigger_type, trigger_id = trigger.split(":")

            args = [
                {"role": arg_role, "id": arg_id}
                for arg_role, arg_id in (arg.split(":") for arg in args)
            ]

            events[event_id] = {
                "id": event_id,
                "trigger_type": trigger_type,
                "trigger_id": trigger_id,
                "args": args,
            }
        elif line.startswith("M") or line.startswith("A"):
            modal_id, modal_type, *reference_ids = line.split()
            modalities[modal_id] = {
                "id": modal_id,
                "type": modal_type,
                "reference_ids": reference_ids,
            }
        elif line.startswith("N") or line.startswith("#"):
            attribute_id, attribute_value = line.split("\t", 1)
            attributes[attribute_id] = {
                "id": attribute_id.strip(),
                "value": attribute_value.strip(),
            }
        elif line.startswith("*"):
            _, relation_tag, *args = line.split()
            if relation_tag == "Equiv":  # Equivalence relations
                equivalences.append(set(args))
            else:
                print(
                    "## Unexpected annotation found: {} in {}".format(
                        line, standoff_file
                    )
                )
        else:
            print(
                "## Unexpected annotation found: {} in {}".format(line, standoff_file)
            )

    return reference, entities, relations, events, modalities, attributes, equivalences


def write_ann_file(
        ann_file,
        entities=None,
        equivalences=None,
        relations=None,
        events=None,
        modalities=None,
        attributes=None,
        normalise_triggers=False,
):
    lines = []

    def _normalise_triggers(s):
        if normalise_triggers:
            return regex.sub(r"^TR", "T", s)
        return s

    if equivalences:
        for equivalence in sorted(
                set(
                    tuple(sorted(equivalence, key=lambda val: int(val.lstrip("TR"))))
                    for equivalence in equivalences
                )
        ):
            if equivalence:
                lines.append(
                    "*\tEquiv " + " ".join(map(_normalise_triggers, equivalence))
                )

    if entities:
        for k in sorted(entities, key=lambda val: int(val.lstrip("TR"))):
            lines.append(
                "{}\t{} {} {}\t{}".format(
                    _normalise_triggers(entities[k]["id"]),
                    entities[k]["type"],
                    entities[k]["start"],
                    entities[k]["end"],
                    entities[k]["ref"],
                )
            )

    if relations:
        for k in sorted(relations, key=lambda val: int(val.lstrip("R"))):
            lines.append(
                "{}\t{} {}:{} {}:{}".format(
                    relations[k]["id"],
                    relations[k]["role"],
                    relations[k]["left_arg"]["label"],
                    _normalise_triggers(relations[k]["left_arg"]["id"]),
                    relations[k]["right_arg"]["label"],
                    _normalise_triggers(relations[k]["right_arg"]["id"]),
                )
            )

    if events:
        for k in sorted(events, key=lambda val: int(val.lstrip("E"))):
            event_annotation = "{}\t{}:{}".format(
                events[k]["id"],
                events[k]["trigger_type"],
                _normalise_triggers(events[k]["trigger_id"]),
            )
            for arg in events[k]["args"]:
                event_annotation += " {}:{}".format(
                    arg["role"], _normalise_triggers(arg["id"])
                )
            lines.append(event_annotation)

    if attributes:
        for k in sorted(attributes, key=lambda val: (val[:1], int(val[1:]))):
            lines.append("{}\t{}".format(attributes[k]["id"], attributes[k]["value"]))

    if modalities:
        for k in sorted(modalities, key=lambda val: int(val.lstrip("M"))):
            lines.append(
                "{}\t{} {}".format(
                    modalities[k]["id"],
                    modalities[k]["type"],
                    " ".join(modalities[k]["reference_ids"]),
                )
            )

    write_lines(lines, ann_file)


def split_token(token, offsets):
    if len(offsets) == 0:
        return [token]

    subtokens = []
    subtokens.append(token[: offsets[0]])

    for i in range(1, len(offsets)):
        subtokens.append(token[offsets[i - 1]: offsets[i]])

    subtokens.append(token[offsets[-1]:])
    return list(filter(len, subtokens))


def extend_offset(offset, doc, reverse=False):
    if reverse:
        while offset < len(doc) and regex.match(r"[^\W_]", doc[offset]):
            offset += 1
    else:
        while offset > 0 and regex.match(r"[^\W_]", doc[offset - 1]):
            offset -= 1
    return offset


def correct_sentence_boundaries_ace05(doc):
    doc = regex.sub(r"(?<!<[^>]+>\s*)\n(?!\s*<[^>]+>)", " ", doc)
    doc = regex.sub(r"<[^>]+>", "", doc)
    return doc


def parse_xml_file_ace05(xml_file, encoding=None):
    assert os.path.exists(xml_file), "XML file not found: " + xml_file

    entities = OrderedDict()
    relations = OrderedDict()
    events = OrderedDict()
    modalities = OrderedDict()
    attributes = OrderedDict()
    equivalences = []

    markup = BeautifulSoup(read_file(xml_file), "lxml-xml")

    source_file_tag = markup.find("source_file")

    # Using reference for double-check
    source_file = os.path.join(os.path.dirname(xml_file), source_file_tag["URI"])
    original_reference = read_text(source_file, encoding=source_file_tag["ENCODING"])
    reference = correct_sentence_boundaries_ace05(original_reference)

    for entity_tag in markup.find_all("entity"):
        for entity_mention_tag in entity_tag.find_all("entity_mention"):
            charseq = entity_mention_tag.find("head").find("charseq")

            entity_id = "T" + entity_mention_tag["ID"].strip()
            entity_type = entity_tag["TYPE"].strip()
            start_offset = int(charseq["START"])
            end_offset = int(charseq["END"]) + 1
            actual_reference = reference[start_offset:end_offset]

            assert charseq.text.replace("\n", " ") == unescape(actual_reference)

            assert entity_id not in entities

            entities[entity_id] = {
                "id": entity_id,
                "type": entity_type,
                "start": start_offset,
                "end": end_offset,
                "ref": actual_reference,
            }

    for value_tag in markup.find_all("value"):
        for value_mention_tag in value_tag.find_all("value_mention"):
            assert value_mention_tag.find("head") is None

            charseq = value_mention_tag.find("extent").find("charseq")

            entity_id = "T" + value_mention_tag["ID"].strip()
            entity_type = (value_tag.get("SUBTYPE") or value_tag["TYPE"]).strip()
            start_offset = int(charseq["START"])
            end_offset = int(charseq["END"]) + 1
            actual_reference = reference[start_offset:end_offset]

            assert charseq.text.replace("\n", " ") == unescape(actual_reference)

            assert entity_id not in entities

            entities[entity_id] = {
                "id": entity_id,
                "type": entity_type,
                "start": start_offset,
                "end": end_offset,
                "ref": actual_reference,
            }

    for timex2_tag in markup.find_all("timex2"):
        for timex2_mention_tag in timex2_tag.find_all("timex2_mention"):
            assert timex2_mention_tag.find("head") is None

            charseq = timex2_mention_tag.find("extent").find("charseq")

            entity_id = "T" + timex2_mention_tag["ID"].strip()
            entity_type = "TIME"
            start_offset = int(charseq["START"])
            end_offset = int(charseq["END"]) + 1
            actual_reference = reference[start_offset:end_offset]

            assert charseq.text.replace("\n", " ") == unescape(actual_reference)

            assert entity_id not in entities

            entities[entity_id] = {
                "id": entity_id,
                "type": entity_type,
                "start": start_offset,
                "end": end_offset,
                "ref": actual_reference,
            }

    for event_tag in markup.find_all("event"):
        for event_mention_tag in event_tag.find_all("event_mention"):
            charseq = event_mention_tag.find("anchor").find("charseq")

            event_id = "E" + event_mention_tag["ID"].strip()

            trigger_id = "T" + event_mention_tag["ID"].strip()
            trigger_type = event_tag["SUBTYPE"].strip()
            start_offset = int(charseq["START"])
            end_offset = int(charseq["END"]) + 1
            actual_reference = reference[start_offset:end_offset]

            assert charseq.text.replace("\n", " ") == unescape(actual_reference)

            assert trigger_id not in entities

            entities[trigger_id] = {
                "id": trigger_id,
                "type": trigger_type,
                "start": start_offset,
                "end": end_offset,
                "ref": actual_reference,
            }

            args = [
                {
                    "role": event_mention_argument_tag["ROLE"],
                    "id": "T" + event_mention_argument_tag["REFID"],
                }
                for event_mention_argument_tag in event_mention_tag.find_all(
                    "event_mention_argument"
                )
            ]

            assert event_id not in events

            events[event_id] = {
                "id": event_id,
                "trigger_type": trigger_type,
                "trigger_id": trigger_id,
                "args": args,
            }

    return reference, entities, relations, events, modalities, attributes, equivalences


def build_subtoken_map(corpus_name, corpus_dir, output_dir):
    subtoken_map = defaultdict(list)

    for fn in glob(os.path.join(corpus_dir, "**/*.txt"), recursive=True):
        entities = OrderedDict()

        basename, _ = os.path.splitext(fn)

        a1_file = basename + ".a1"
        if os.path.exists(a1_file):
            original_doc, a1_entities, *_ = parse_standoff_file(
                a1_file, fn, encoding="UTF-8"
            )
            entities.update(a1_entities)
        else:
            print("A1 file missing: " + a1_file)

        a2_file = basename + ".a2"
        if os.path.exists(a2_file):
            _, a2_entities, *_ = parse_standoff_file(a2_file, fn, encoding="UTF-8")
            entities.update(a2_entities)
        else:
            print("A2 file missing: " + a2_file)

        for entity in entities.values():
            entity_start, entity_end = entity["start"], entity["end"]

            while original_doc[entity_start] == " ":
                entity_start += 1

            while original_doc[entity_end - 1] == " ":
                entity_end -= 1

            extended_entity_start = extend_offset(
                entity_start, original_doc, reverse=False
            )
            extended_entity_end = extend_offset(entity_end, original_doc, reverse=True)

            reference_string = original_doc[extended_entity_start:extended_entity_end]

            assert reference_string.strip() == reference_string

            if extended_entity_start < entity_start or entity_end < extended_entity_end:
                full_token = original_doc[
                             extended_entity_start:extended_entity_end
                             ].lower()

                unique_offsets = sorted(
                    {
                        entity_start,
                        entity_end,
                        extended_entity_start,
                        extended_entity_end,
                    }
                )  # > 2

                offsets = [
                    unique_offsets[i] - unique_offsets[0]
                    for i in range(1, len(unique_offsets) - 1)
                ]

                subtoken_map[full_token].extend(offsets)

    subtoken_map_fn = os.path.join(output_dir, corpus_name + ".subtoken.map")
    if os.path.exists(subtoken_map_fn):
        for k, v in read_json(subtoken_map_fn).items():
            subtoken_map[k].extend(v)

    ordered_subtoken_map = OrderedDict()

    for k in sorted(subtoken_map, key=len):
        ordered_subtoken_map[k] = sorted(set(subtoken_map[k]))

    write_json(ordered_subtoken_map, subtoken_map_fn)


def convert(corpus_name, corpus_dir, output_dir):
    defined_types = defaultdict(set)

    for fn in glob(os.path.join(corpus_dir, "**/*.txt"), recursive=True):
        basename, _ = os.path.splitext(fn)

        a1_file = basename + ".a1"
        if os.path.exists(a1_file):
            _, _, _, events, _, _, _ = parse_standoff_file(
                a1_file, fn, encoding="UTF-8"
            )
            for event in events.values():
                defined_types["trigger_types"].add(event["trigger_type"])

        a2_file = basename + ".a2"
        if os.path.exists(a2_file):
            _, _, _, events, _, _, _ = parse_standoff_file(
                a2_file, fn, encoding="UTF-8"
            )
            for event in events.values():
                defined_types["trigger_types"].add(event["trigger_type"])

    subtoken_map = []

    subtoken_map_fn = os.path.join(output_dir, corpus_name + ".subtoken.map")

    if os.path.exists(subtoken_map_fn):
        print(">> Loading subtoken map...")
        _subtoken_map = read_json(subtoken_map_fn)
        for pattern in sorted(_subtoken_map, key=len):
            subtoken_map.append((pattern, _subtoken_map[pattern]))

    for fn in glob(os.path.join(corpus_dir, "**/*.txt"), recursive=True):
        print(">> Processing: " + fn)

        entities = OrderedDict()
        relations = OrderedDict()
        events = OrderedDict()
        modalities = OrderedDict()
        attributes = OrderedDict()
        equivalences = []

        a1_entities = None
        a1_relations = None
        a1_events = None
        a1_modalities = None
        a1_attributes = None
        a1_equivalences = None

        a2_entities = None
        a2_relations = None
        a2_events = None
        a2_modalities = None
        a2_attributes = None
        a2_equivalences = None

        basename, _ = os.path.splitext(fn)

        a1_file = basename + ".a1"
        if os.path.exists(a1_file):
            _, a1_entities, a1_relations, a1_events, a1_modalities, a1_attributes, a1_equivalences = parse_standoff_file(
                a1_file, fn, encoding="UTF-8"
            )
            entities.update(a1_entities)
            relations.update(a1_relations)
            events.update(a1_events)
            modalities.update(a1_modalities)
            attributes.update(a1_attributes)
            equivalences.extend(a1_equivalences)
        else:
            print("A1 file missing: " + a1_file)

        a2_file = basename + ".a2"
        if os.path.exists(a2_file):
            _, a2_entities, a2_relations, a2_events, a2_modalities, a2_attributes, a2_equivalences = parse_standoff_file(
                a2_file, fn, encoding="UTF-8"
            )
            entities.update(a2_entities)
            relations.update(a2_relations)
            events.update(a2_events)
            modalities.update(a2_modalities)
            attributes.update(a2_attributes)
            equivalences.extend(a2_equivalences)
        else:
            print("A2 file missing: " + a2_file)

        doc_fn = os.path.join(
            output_dir, corpus_name, os.path.relpath(basename, corpus_dir)
        )

        original_doc = read_text(fn)

        write_text(original_doc, doc_fn + ".txt.ori")

        cursor = 0
        offset_map = {}
        sentence_boundaries = []

        # Split into sentences and ensure that there is no broken entities
        for sentence_idx, (start_offset, end_offset) in enumerate(
                generate_sentence_boundaries(original_doc)
        ):
            sentence_boundaries.append({"start": start_offset, "end": end_offset})
            for offset in range(start_offset, end_offset + 1):
                offset_map[offset] = {"offset": cursor, "line": sentence_idx}
                cursor += 1  # This will include the newline at the end of sentence

        # Correct broken sentence boundaries
        for entity in entities.values():
            entity_start, entity_end = entity["start"], entity["end"]

            while original_doc[entity_start] == " ":
                entity_start += 1

            while original_doc[entity_end - 1] == " ":
                entity_end -= 1

            left_line_idx = offset_map[entity_start]["line"]
            right_line_idx = offset_map[entity_end]["line"]

            if left_line_idx != right_line_idx:
                sentence_boundaries[min(left_line_idx, right_line_idx)]["broken"] = max(
                    sentence_boundaries[min(left_line_idx, right_line_idx)].get(
                        "broken", -1
                    ),
                    left_line_idx,
                    right_line_idx,
                )

        # Merge broken sentences into a sentence
        sentence_idx = 0
        normalised_sentences = []

        while sentence_idx < len(sentence_boundaries):
            start_offset = sentence_boundaries[sentence_idx]["start"]
            end_offset = sentence_boundaries[sentence_idx]["end"]

            while (
                    sentence_idx < len(sentence_boundaries)
                    and "broken" in sentence_boundaries[sentence_idx]
            ):
                broken_sentence_idx = sentence_boundaries[sentence_idx]["broken"]
                end_offset = sentence_boundaries[broken_sentence_idx]["end"]
                sentence_idx = broken_sentence_idx

            normalised_sentences.append(original_doc[start_offset:end_offset])
            sentence_idx += 1

        # Normalize special tokens
        sentences = []

        for sentence in normalised_sentences:
            for pattern, split_points in subtoken_map:
                matcher = regex.search(
                    r"\b{}\b".format(regex.escape(pattern)),
                    sentence,
                    flags=regex.IGNORECASE,
                )
                while matcher:
                    normalized_span = " ".join(
                        split_token(matcher.group(0), split_points)
                    )

                    sentence = (
                            sentence[: matcher.start()]
                            + normalized_span
                            + sentence[matcher.end():]
                    )

                    matcher = regex.search(
                        r"\b{}\b".format(regex.escape(pattern)),
                        sentence,
                        flags=regex.IGNORECASE,
                    )

            tokens = []

            for token in sentence.split():
                for subtoken in BASIC_TOKENIZER.tokenize(token):
                    tokens.append(subtoken)

            sentences.append(" ".join(tokens))

        normalized_doc = "\n".join(sentences)

        print(">> Building offset map...")

        # Build offset map
        offset_map = {}
        inverse_offset_map = {}

        original_doc_pos = 0
        normalized_doc_pos = 0

        _original_doc = regex.sub(
            r"\s", " ", original_doc
        )  # Address special blank characters
        _normalized_doc = normalized_doc.replace("\r", " ").replace("\n", " ")

        while original_doc_pos < len(_original_doc) and normalized_doc_pos < len(
                _normalized_doc
        ):
            original_doc_char = _original_doc[original_doc_pos]
            normalized_doc_char = _normalized_doc[normalized_doc_pos]

            if original_doc_char == normalized_doc_char:
                offset_map[original_doc_pos] = normalized_doc_pos
                inverse_offset_map[normalized_doc_pos] = original_doc_pos
                original_doc_pos += 1
                normalized_doc_pos += 1
            else:
                if original_doc_char == " ":
                    offset_map[original_doc_pos] = normalized_doc_pos
                    original_doc_pos += 1
                elif normalized_doc_char == " ":
                    inverse_offset_map[normalized_doc_pos] = original_doc_pos
                    normalized_doc_pos += 1

        if offset_map:
            offset_map[max(offset_map) + 1] = max(offset_map.values()) + 1

        if inverse_offset_map:
            inverse_offset_map[max(inverse_offset_map) + 1] = (
                    max(inverse_offset_map.values()) + 1
            )

        assert max(offset_map.values()) == len(_normalized_doc) and max(
            inverse_offset_map
        ) == len(
            _normalized_doc
        )  # To ensure the code above is right

        write_json(offset_map, doc_fn + ".map")

        write_text(normalized_doc, doc_fn + ".txt")

        print(">> Generating annotation file...")

        for entity in entities.values():
            entity_start, entity_end = entity["start"], entity["end"]

            while original_doc[entity_start] == " ":
                entity_start += 1

            while original_doc[entity_end - 1] == " ":
                entity_end -= 1

            entity_start_offset = offset_map[entity_start]
            entity_end_offset = offset_map[entity_end]

            while normalized_doc[entity_start_offset] == " ":
                entity_start_offset += 1

            while normalized_doc[entity_end_offset - 1] == " ":
                entity_end_offset -= 1

            normalized_entity = normalized_doc[entity_start_offset:entity_end_offset]

            assert normalized_entity.strip() == normalized_entity and regex.sub(
                r"\s+", "", entity["ref"]
            ) == regex.sub(
                r" +", "", normalized_entity
            )  # Thank God

            entity_id = entity["id"]

            inverse_offset_map.setdefault("entities", {})[entity_id] = (
                entity["start"],
                entity["end"],
            )

            entity["alt_id"] = entity_id
            entity["start"] = entity_start_offset
            entity["end"] = entity_end_offset
            entity["ref"] = normalized_entity

            if entity["type"] in defined_types["trigger_types"]:
                entity["alt_id"] = regex.sub(r"^T(?!R)", "TR", entity_id)
            else:
                defined_types["entity_types"].add(entity["type"])

            for _entities in (a1_entities, a2_entities):
                if _entities and entity_id in _entities:
                    _entities[entity_id] = entity

        write_json(inverse_offset_map, doc_fn + ".inv.map")

        write_ann_file(
            doc_fn + ".a1",
            entities=a1_entities,
            equivalences=a1_equivalences,
            relations=a1_relations,
            events=a1_events,
            modalities=a1_modalities,
            attributes=a1_attributes,
        )

        write_ann_file(
            doc_fn + ".a2",
            entities=a2_entities,
            equivalences=a2_equivalences,
            relations=a2_relations,
            events=a2_events,
            modalities=a2_modalities,
            attributes=a2_attributes,
        )

        relation_pairs = set()

        for event in events.values():
            trigger_id = event["trigger_id"]
            trigger_id = entities[trigger_id]["alt_id"]
            event["trigger_id"] = trigger_id

            for arg in event["args"]:
                arg_id = arg["id"]
                if arg_id in events:
                    arg_id = events[arg_id]["trigger_id"]
                if arg_id in entities:
                    arg_id = entities[arg_id]["alt_id"]

                relation_pairs.add((trigger_id, arg_id, arg["role"]))

        for entity in entities.values():
            entity["id"] = entity["alt_id"]

        relation_count = 1

        for trigger_id, arg_id, relation_role in relation_pairs:
            relation_id = "R" + str(relation_count)

            relation_role = relation_role.strip(string.digits)

            relations[relation_id] = {
                "id": relation_id,
                "role": relation_role,
                "left_arg": {"label": "Arg1", "id": trigger_id},
                "right_arg": {"label": "Arg2", "id": arg_id},
            }

            relation_count += 1

            defined_types["relation_types"].add(relation_role)

        write_ann_file(
            doc_fn + ".ann",
            entities=entities,
            equivalences=equivalences,
            relations=relations,
            events=events,
            modalities=modalities,
            attributes=attributes,
        )

    defined_types = {k: sorted(v) for k, v in defined_types.items()}
    write_json(defined_types, os.path.join(output_dir, corpus_name + ".types"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, required=True, help='--indir')
    parser.add_argument('--outdir', type=str, required=True, help='--outdir')
    args = parser.parse_args()

    input_dir = getattr(args, 'indir')
    output_dir = getattr(args, 'outdir')

    # Step 1:
    # Collect token which contain entity as substring
    for fn in glob(os.path.join(input_dir, "*")):
        if os.path.isdir(fn):
            print("Building subtoken map: " + os.path.basename(fn))
            build_subtoken_map(os.path.relpath(fn, input_dir), fn, output_dir)

    time.sleep(10)

    # Step 2:
    # Build annotation files
    for fn in glob(os.path.join(input_dir, "*")):
        if os.path.isdir(fn):
            print("Converting: " + os.path.basename(fn))
            convert(os.path.relpath(fn, input_dir), fn, output_dir)


if __name__ == "__main__":
    main()
