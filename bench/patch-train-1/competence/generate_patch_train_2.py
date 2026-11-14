#!/usr/bin/env python3
"""PATCH-TRAIN-2 dataset generator (DRAFT; see PATCH-TRAIN-2-DATA-PLAN.md). Deterministic, runs NO model, only the tokenizer.

Splits (entity-disjoint from each other and from the discovery files): cal (4 pairs), conf (12 pairs), ret (24 retention pool).
    python3 -I generate_patch_train_2.py --split cal --dummy-seed 7 --out-dir DIR       (dummy: any seed except the real ones)
    python3 -I generate_patch_train_2.py --split cal --real                              (real seeds; guarded)
Real generation refuses unless this file and patch_train_2_seeds.json are committed and unmodified, and the split is not already written;
`conf` additionally needs --frozen-protocol PATH to an existing file. The generator never reads or writes a model."""
import argparse
import hashlib
import json
import os
import random
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from build_tasks import NEUTRAL_TEXT, REAL_FACTS, REPO, REVISION, SYLLABLES, TASKS  # noqa: E402  (constants only; build_tasks is unchanged)

SCHEMA = "patch-train-2-tasks/1"
SEEDS_FILE = os.path.join(HERE, "patch_train_2_seeds.json")
DISCOVERY_FILES = ("tasks_discovery.json", "tasks.json")
TOKEN_RANGE = (2, 6)            # tokens of " " + name; the range spanned by every discovery name (plan §2)
RET_PROMPT_ENTITIES = 23        # RET = {x_A} + the first 23 RET-split entities' training prompts (plan §2.2)
SPLITS = {"cal": {"pairs": 4, "paired": True}, "conf": {"pairs": 12, "paired": True}, "ret": {"entities": 24, "paired": False}}
ANSWERS = TASKS["A"]["answers"]
TRAIN_TEMPLATE = TASKS["A"]["train_templates"][0]
HELDOUT_TEMPLATES = TASKS["A"]["heldout_templates"]
OUT_NAME = "patch_train_2_{split}.json"


def n_entities(split):
    spec = SPLITS[split]
    return spec["pairs"] * 2 if spec["paired"] else spec["entities"]


def sha256_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def file_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def discovery_names(directory=HERE):
    """Every entity name (both tasks) of both discovery files, plus the real-fact countries."""
    names = {c for c, _ in REAL_FACTS}
    for fname in DISCOVERY_FILES:
        with open(os.path.join(directory, fname)) as f:
            doc = json.load(f)
        for t in doc["tasks"].values():
            names.update(t["entities"])
    return names


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def nearest(name, others):
    best = min(others, key=lambda o: (edit_distance(name, o), o))
    return {"name": best, "normalised_edit_distance": round(edit_distance(name, best) / max(len(name), len(best)), 4)}


def draw_name(rng):
    """The discovery syllable process (build_tasks.names, task A): 2-3 syllables, capitalised, suffix 'ia'."""
    return "".join(rng.choice(SYLLABLES) for _ in range(rng.choice([2, 3]))).capitalize() + "ia"


def generate_names(rng, count, excluded, token_count):
    seen, out = set(excluded), []
    while len(out) < count:
        name = draw_name(rng)
        if name in seen or not (TOKEN_RANGE[0] <= token_count(name) <= TOKEN_RANGE[1]):
            continue            # rejected by format/uniqueness rule only; no model quantity
        seen.add(name)
        out.append(name)
    return out


def assign_answers(rng, split):
    """Deterministic answer balancing (plan §2.1). No model quantity, no rejection, no redraw."""
    n = n_entities(split)
    if split == "cal":
        return rng.sample(ANSWERS, 12)[:n]                 # each answer at most once -> pair members differ
    if split == "conf":
        p = rng.sample(ANSWERS, 12)
        k = rng.randint(1, 11)
        out = []
        for i in range(12):
            out += [p[i], p[(i + k) % 12]]                 # each answer once per column -> exactly twice; k != 0 -> members differ
        return out
    return rng.sample(ANSWERS, 12) + rng.sample(ANSWERS, 12)   # ret: each answer exactly twice, unpaired


def generate_split(split, seed, excluded, token_count, provenance):
    rng = random.Random(seed)
    n = n_entities(split)
    names = generate_names(rng, n, excluded, token_count)
    answers = assign_answers(rng, split)
    assert len(names) == len(answers) == n
    disc = sorted(discovery_names())
    other_split = sorted(set(excluded) - set(disc))
    entities = []
    for j, (name, ans) in enumerate(zip(names, answers)):
        e = {"index": j, "name": name, "answer": ans, "token_count": token_count(name),
             "nearest_discovery_name": nearest(name, disc)}
        if other_split:
            e["nearest_other_split_name"] = nearest(name, other_split)
        if SPLITS[split]["paired"]:
            e["pair"] = j // 2
        entities.append(e)
    doc = {"schema": SCHEMA, "split": split, "seed": seed, **provenance, "relation": TASKS["A"]["relation"],
           "train_template": TRAIN_TEMPLATE, "heldout_templates": HELDOUT_TEMPLATES, "answers": ANSWERS,
           "token_range": list(TOKEN_RANGE), "entities": entities, "real_facts": [{"country": c, "answer": a, "template": TRAIN_TEMPLATE}
                                                                                    for c, a in REAL_FACTS],
           "neutral_text": NEUTRAL_TEXT}
    if SPLITS[split]["paired"]:
        doc["pairs"] = [{"pair": i, "entities": [2 * i, 2 * i + 1], "primary_first": 2 * i + rng.randint(0, 1)} for i in range(n // 2)]
    else:
        doc["retention_prompt_entities"] = list(range(RET_PROMPT_ENTITIES))
        doc["held_back_leakage_entity"] = RET_PROMPT_ENTITIES
    return doc


def retention_prompts(ret_doc, first_patch_prompt):
    """RET = {x_A} + the first 23 RET-split entities' TRAINING-template prompts (plan §2.2): exactly 24 prompts."""
    ents = ret_doc["entities"]
    prompts = [first_patch_prompt] + [ret_doc["train_template"].format(e=ents[i]["name"]) for i in ret_doc["retention_prompt_entities"]]
    assert len(prompts) == 1 + RET_PROMPT_ENTITIES == 24
    return prompts


def uncovered_unrelated(pair_doc, ret_doc, pair_index):
    """Never-patched entity names the X gates use (plan §2.3): every entity of the other pairs of the same split, plus the held-back
    RET-split entity. The pair's own partner is added by the harness at the stages where it is still unpatched. The 23 RET-covered
    entities are deliberately excluded (distillation optimises them)."""
    members = set(pair_doc["pairs"][pair_index]["entities"])
    return [e["name"] for e in pair_doc["entities"] if e["index"] not in members] + \
           [ret_doc["entities"][ret_doc["held_back_leakage_entity"]]["name"]]


def make_token_counter():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(REPO, revision=REVISION, local_files_only=True)
    return lambda name: len(tok(" " + name, add_special_tokens=False)["input_ids"]), tok


def check_answers_single_token(tok):
    bad = [a for a in ANSWERS if len(tok(a, add_special_tokens=False)["input_ids"]) != 1]
    assert not bad, f"multi-token answers {bad}"


def load_seeds():
    with open(SEEDS_FILE) as f:
        doc = json.load(f)
    return {k: v for k, v in doc.items() if isinstance(v, int)}     # cal / conf / ret only


def read_split(path):
    with open(path) as f:
        return json.load(f)


def git_clean_and_committed(paths):
    top = subprocess.run(["git", "-C", HERE, "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True).stdout.strip()
    for p in paths:
        rel = os.path.relpath(p, top)
        if subprocess.run(["git", "-C", top, "ls-files", "--error-unmatch", rel], capture_output=True).returncode != 0:
            return False, f"{rel} is not committed"
        if subprocess.run(["git", "-C", top, "diff", "--quiet", "HEAD", "--", rel], capture_output=True).returncode != 0:
            return False, f"{rel} differs from HEAD"
    return True, ""


def split_exclusions(split, directory):
    """Splits generated earlier must be excluded by the later ones: cal <- discovery; ret <- + cal; conf <- + cal + ret."""
    names = discovery_names()
    for earlier in {"cal": [], "ret": ["cal"], "conf": ["cal", "ret"]}[split]:
        path = os.path.join(directory, OUT_NAME.format(split=earlier))
        if not os.path.exists(path):
            raise SystemExit(f"split '{earlier}' must be generated and present in {directory} before '{split}'")
        names.update(e["name"] for e in read_split(path)["entities"])
    return names


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", required=True, choices=sorted(SPLITS))
    ap.add_argument("--dummy-seed", type=int)
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--out-dir", default=HERE)
    ap.add_argument("--frozen-protocol")
    a = ap.parse_args(argv)
    if bool(a.real) == (a.dummy_seed is not None):
        raise SystemExit("give exactly one of --real or --dummy-seed")
    seeds = load_seeds()
    if a.real:
        ok, why = git_clean_and_committed([os.path.abspath(__file__), SEEDS_FILE])
        if not ok:
            raise SystemExit(f"refusing real generation: {why}")
        if a.split == "conf" and not (a.frozen_protocol and os.path.exists(a.frozen_protocol)):
            raise SystemExit("refusing real conf generation: --frozen-protocol PATH to an existing frozen protocol is required")
        seed, dummy = seeds[a.split], False
        if os.path.abspath(a.out_dir) != HERE:
            raise SystemExit("real splits are written next to the generator")
    else:
        if a.dummy_seed in seeds.values():
            raise SystemExit("a dummy seed must not be a real seed")
        seed, dummy = a.dummy_seed, True
    out_path = os.path.join(a.out_dir, OUT_NAME.format(split=a.split))
    if os.path.exists(out_path):
        raise SystemExit(f"{out_path} exists; splits are never regenerated over")
    token_count, tok = make_token_counter()
    check_answers_single_token(tok)
    excluded = split_exclusions(a.split, a.out_dir)
    provenance = {"dummy": dummy, "generator_sha256": file_sha256(os.path.abspath(__file__)), "seeds_file_sha256": file_sha256(SEEDS_FILE),
                  "exclusion_sha256": sha256_text("\n".join(sorted(excluded))), "exclusion_count": len(excluded),
                  "tokenizer": {"repo": REPO, "revision": REVISION}}
    doc = generate_split(a.split, seed, excluded, token_count, provenance)
    os.makedirs(a.out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    print(f"wrote {out_path}: {len(doc['entities'])} entities, dummy={dummy}")


if __name__ == "__main__":
    main()
