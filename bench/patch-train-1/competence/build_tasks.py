#!/usr/bin/env python3
"""Build the candidate PATCH-TRAIN-1 task set (tasks.json), deterministically. Baseline design only; nothing is trained.

Two independent factual-association tasks about FICTIONAL entities (so the base model cannot already know them), a
real-fact set to check the prompt formats work, and collateral sets. Every answer is one token (with its leading space), so
a prompt's loss is a single cross-entropy. Run: python3 build_tasks.py  (writes tasks.json; needs the HF tokenizer cache).
"""
import json
import os
import random
import sys

from transformers import AutoTokenizer

REPO, REVISION = "google/gemma-3-270m-it", "ac82b4e820549b854eebf28ce6dedaf9fdfa17b3"
SEED = 20261112
N_ENTITIES = 24
SYLLABLES = ["zor", "van", "quel", "bri", "tha", "mok", "lun", "dre", "pyl", "gar", "nex", "vol", "kir", "sal", "yem", "fro"]

TASKS = {
    "A": {"relation": "capital of a fictional country",
          "entity_kind": "country",
          "answers": [" Lyon", " Leeds", " Bergen", " Porto", " Turin", " Dallas", " Denver", " Austin", " Boston",
                      " Seattle", " Nagoya", " Osaka"],
          "train_templates": ["The capital of {e} is"],
          "heldout_templates": ["{e}'s capital city is", "The capital city of {e} is called", "Q: What is the capital of {e}? A:"]},
    "B": {"relation": "occupation of a fictional person",
          "entity_kind": "person",
          "answers": [" doctor", " pilot", " farmer", " teacher", " lawyer", " baker", " painter", " sailor", " nurse",
                      " chef", " miner", " judge"],
          "train_templates": ["The occupation of {e} is"],
          "heldout_templates": ["{e}'s job is", "{e}'s profession is", "Q: What does {e} do for a living? A:"]},
}
REAL_FACTS = [("France", " Paris"), ("Italy", " Rome"), ("Spain", " Madrid"), ("Germany", " Berlin"), ("Japan", " Tokyo"),
              ("Egypt", " Cairo"), ("Russia", " Moscow"), ("Greece", " Athens"), ("Austria", " Vienna"),
              ("Portugal", " Lisbon"), ("Ireland", " Dublin"), ("Norway", " Oslo"), ("Sweden", " Stockholm"),
              ("Poland", " Warsaw"), ("Cuba", " Havana"), ("Kenya", " Nairobi"), ("Peru", " Lima"), ("Chile", " Santiago"),
              ("Turkey", " Ankara"), ("Hungary", " Budapest"), ("Finland", " Helsinki"), ("Belgium", " Brussels"),
              ("Iran", " Tehran"), ("Denmark", " Copenhagen")]
NEUTRAL_TEXT = ["The river runs slowly through the old town before it reaches the sea.",
                "She opened the window and listened to the rain falling on the roof.",
                "In 1969 the first humans walked on the surface of the Moon.",
                "To make bread you need flour, water, salt and a little yeast.",
                "The committee will meet again on Thursday to discuss the budget.",
                "Light travels faster than sound, which is why lightning appears before thunder.",
                "He packed a small bag, locked the door and walked to the station.",
                "A triangle has three sides and the sum of its angles is one hundred and eighty degrees."]


def names(rng, count, kind, taken=frozenset()):
    seen, out = set(taken), []
    while len(out) < count:
        syl = [rng.choice(SYLLABLES) for _ in range(rng.choice([2, 3]))]
        word = "".join(syl).capitalize()
        name = word + ("ia" if kind == "country" else "") if kind == "country" else word + " " + rng.choice(SYLLABLES).capitalize() + rng.choice(["an", "or", "el", "us"])
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    opt = {"--seed": SEED, "--out": "tasks.json", "--exclude": None}
    for flag in opt:
        if flag in args:
            opt[flag] = args[args.index(flag) + 1]
    seed, out_name = int(opt["--seed"]), opt["--out"]
    taken = set()
    if opt["--exclude"]:   # entity names of another set that this one must not reuse (discovery vs confirmatory)
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), opt["--exclude"])) as f:
            other = json.load(f)
        taken = {e for t in other["tasks"].values() for e in t["entities"]}
    tok = AutoTokenizer.from_pretrained(REPO, revision=REVISION, local_files_only=True)

    def single(answer):
        ids = tok(answer, add_special_tokens=False)["input_ids"]
        return len(ids) == 1

    rng = random.Random(seed)
    doc = {"schema": "patch-train-1-candidate-tasks/1", "seed": seed, "model": {"repo": REPO, "revision": REVISION},
           "tasks": {}, "real_facts": [], "neutral_text": NEUTRAL_TEXT, "excluded": {}}
    for key, t in TASKS.items():
        bad = [a for a in t["answers"] if not single(a)]
        assert not bad, f"task {key}: multi-token answers {bad}"
        ents = names(rng, N_ENTITIES, t["entity_kind"], taken)
        order = list(range(len(ents)))
        rng.shuffle(order)
        # entities 0..15 are TRAINED, 16..23 are held-out ENTITIES (never trained; same relation, same templates)
        assigned = {e: t["answers"][i % len(t["answers"])] for i, e in enumerate(ents)}
        doc["tasks"][key] = {**t, "entities": ents, "assignment": assigned,
                             "trained_entities": ents[:16], "heldout_entities": ents[16:]}
    keep = [(c, a) for c, a in REAL_FACTS if single(a)]
    doc["excluded"]["real_facts_multi_token_answer"] = [c for c, a in REAL_FACTS if not single(a)]
    doc["real_facts"] = [{"country": c, "answer": a, "template": "The capital of {e} is"} for c, a in keep]
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), out_name), "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    print({k: (len(v["trained_entities"]), len(v["heldout_entities"])) for k, v in doc["tasks"].items()},
          "real facts kept:", len(keep), "excluded:", doc["excluded"])


if __name__ == "__main__":
    main()
