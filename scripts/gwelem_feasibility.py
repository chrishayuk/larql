"""GW-ELEMENT-1 feasibility: model-free tokenizer audit for chemical element -> symbol vs atomic number.

    gwelem_feasibility.py OUT_DIR

Owner GO 2026-11-21: model-free feasibility and protocol design only. No model
is loaded and nothing is run. It uses the pinned Gemma 3 4B tokenizer (the
same file and hash as Test 3).

It audits, for every element 1-118 and every candidate phrasing pair:
- **window alignment**: the window is the tokens after the subject up to
  the readout (the final "is"), right-aligned between the two relations as in
  Test 2/3. The windows must have equal length and constant token ids across
  entities, and the relation phrase must lie fully inside the window;
- **answer first tokens**: symbol and atomic number, the full tokenisation,
  the first token, whether it is entity-specific, and its multiplicity;
- **subject echo**: whether the answer's first token occurs in the subject's tokens;
- **subject tokenisation**: token counts, and identical subject ids across the two relations.

It also freezes an entity-disjoint split (salted SHA-256 rank) over all 118
elements *before* any natural baseline, so eligibility can never choose the split.
"""
from __future__ import annotations

import collections
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gwtemplate_baseline_plan as bp  # noqa: E402
import gwtemplate_feasibility as feas  # noqa: E402

# IUPAC names (aluminium, caesium, sulfur). (Z, symbol, name)
ELEMENTS = [
    (1, "H", "hydrogen"), (2, "He", "helium"), (3, "Li", "lithium"), (4, "Be", "beryllium"), (5, "B", "boron"),
    (6, "C", "carbon"), (7, "N", "nitrogen"), (8, "O", "oxygen"), (9, "F", "fluorine"), (10, "Ne", "neon"),
    (11, "Na", "sodium"), (12, "Mg", "magnesium"), (13, "Al", "aluminium"), (14, "Si", "silicon"),
    (15, "P", "phosphorus"), (16, "S", "sulfur"), (17, "Cl", "chlorine"), (18, "Ar", "argon"),
    (19, "K", "potassium"), (20, "Ca", "calcium"), (21, "Sc", "scandium"), (22, "Ti", "titanium"),
    (23, "V", "vanadium"), (24, "Cr", "chromium"), (25, "Mn", "manganese"), (26, "Fe", "iron"),
    (27, "Co", "cobalt"), (28, "Ni", "nickel"), (29, "Cu", "copper"), (30, "Zn", "zinc"), (31, "Ga", "gallium"),
    (32, "Ge", "germanium"), (33, "As", "arsenic"), (34, "Se", "selenium"), (35, "Br", "bromine"),
    (36, "Kr", "krypton"), (37, "Rb", "rubidium"), (38, "Sr", "strontium"), (39, "Y", "yttrium"),
    (40, "Zr", "zirconium"), (41, "Nb", "niobium"), (42, "Mo", "molybdenum"), (43, "Tc", "technetium"),
    (44, "Ru", "ruthenium"), (45, "Rh", "rhodium"), (46, "Pd", "palladium"), (47, "Ag", "silver"),
    (48, "Cd", "cadmium"), (49, "In", "indium"), (50, "Sn", "tin"), (51, "Sb", "antimony"),
    (52, "Te", "tellurium"), (53, "I", "iodine"), (54, "Xe", "xenon"), (55, "Cs", "caesium"),
    (56, "Ba", "barium"), (57, "La", "lanthanum"), (58, "Ce", "cerium"), (59, "Pr", "praseodymium"),
    (60, "Nd", "neodymium"), (61, "Pm", "promethium"), (62, "Sm", "samarium"), (63, "Eu", "europium"),
    (64, "Gd", "gadolinium"), (65, "Tb", "terbium"), (66, "Dy", "dysprosium"), (67, "Ho", "holmium"),
    (68, "Er", "erbium"), (69, "Tm", "thulium"), (70, "Yb", "ytterbium"), (71, "Lu", "lutetium"),
    (72, "Hf", "hafnium"), (73, "Ta", "tantalum"), (74, "W", "tungsten"), (75, "Re", "rhenium"),
    (76, "Os", "osmium"), (77, "Ir", "iridium"), (78, "Pt", "platinum"), (79, "Au", "gold"),
    (80, "Hg", "mercury"), (81, "Tl", "thallium"), (82, "Pb", "lead"), (83, "Bi", "bismuth"),
    (84, "Po", "polonium"), (85, "At", "astatine"), (86, "Rn", "radon"), (87, "Fr", "francium"),
    (88, "Ra", "radium"), (89, "Ac", "actinium"), (90, "Th", "thorium"), (91, "Pa", "protactinium"),
    (92, "U", "uranium"), (93, "Np", "neptunium"), (94, "Pu", "plutonium"), (95, "Am", "americium"),
    (96, "Cm", "curium"), (97, "Bk", "berkelium"), (98, "Cf", "californium"), (99, "Es", "einsteinium"),
    (100, "Fm", "fermium"), (101, "Md", "mendelevium"), (102, "No", "nobelium"), (103, "Lr", "lawrencium"),
    (104, "Rf", "rutherfordium"), (105, "Db", "dubnium"), (106, "Sg", "seaborgium"), (107, "Bh", "bohrium"),
    (108, "Hs", "hassium"), (109, "Mt", "meitnerium"), (110, "Ds", "darmstadtium"), (111, "Rg", "roentgenium"),
    (112, "Cn", "copernicium"), (113, "Nh", "nihonium"), (114, "Fl", "flerovium"), (115, "Mc", "moscovium"),
    (116, "Lv", "livermorium"), (117, "Ts", "tennessine"), (118, "Og", "oganesson"),
]

# Candidate phrasing pairs: id -> {relation: template}. {X} lower-case mid-sentence, capitalised at the start.
CANDIDATES = {
    "A2": {"symbol": "For {x}, the chemical symbol is", "number": "For {x}, the atomic number is"},
    "A3": {"symbol": "{X}'s chemical symbol is", "number": "{X}'s atomic number is"},
    "B2": {"symbol": "For {x}, the symbol is", "number": "For {x}, the number is"},
    "C2": {"symbol": "For the element {x}, the chemical symbol is", "number": "For the element {x}, the atomic number is"},
    "C3": {"symbol": "In chemistry, the symbol for {x} is", "number": "In chemistry, the atomic number of {x} is"},
    "D2": {"symbol": "For {x}, the element symbol is", "number": "For {x}, the proton number is"},
}
RELATION_WORDS = {"symbol": "symbol", "number": "number"}
SPLIT_SALT = "gw-element-1/split/v1|"
SPLIT = (("fit", 0.5), ("development", 0.2), ("confirmation", 0.3))


def answer_text(rel, e):
    return e[1] if rel == "symbol" else str(e[0])


def render(t, name):
    return t.replace("{x}", name).replace("{X}", name[0].upper() + name[1:])


def coords(tok, template, name):
    text = render(template, name)
    subj = name if "{x}" in template else name[0].upper() + name[1:]
    enc = tok.encode(text, add_special_tokens=True)
    s0 = text.index(subj)
    s1 = s0 + len(subj)
    st = bp.span(enc, text, s0, s1)
    return {"text": text, "ids": enc.ids, "subject": st, "readout": len(enc.ids) - 1}


def split_of(z: int) -> dict:
    order = sorted((e[0] for e in ELEMENTS), key=lambda k: hashlib.sha256(f"{SPLIT_SALT}{k}".encode()).hexdigest())
    out, i = {}, 0
    for name, frac in SPLIT:
        n = round(frac * len(order)) if name != SPLIT[-1][0] else len(order) - i
        for k in order[i:i + n]:
            out[k] = name
        i += n
    return out


def main(out: Path) -> int:
    if out.exists():
        raise SystemExit(f"{out} exists")
    out.mkdir(parents=True)
    tok = bp.tokenizer()
    dec = lambda i: tok.decode([i])  # noqa: E731
    rep = {"schema": "larql.gwelem.feasibility.v1", "tokenizer_sha256": "sha256:" + feas.TOKENIZER_SHA,
           "elements": len(ELEMENTS), "model_free": True, "answers": {}, "phrasings": {}}
    # answers
    for rel in ("symbol", "number"):
        firsts = collections.Counter()
        per = {}
        for e in ELEMENTS:
            ids = tok.encode(" " + answer_text(rel, e), add_special_tokens=False).ids
            per[e[0]] = [dec(i) for i in ids]
            firsts[ids[0]] += 1
        single = sum(1 for z, t in per.items() if len(t) == 1)
        rep["answers"][rel] = {
            "first_token_classes": len(firsts), "max_multiplicity": max(firsts.values()),
            "entities_with_unique_first_token": sum(1 for e in ELEMENTS
                                                    if firsts[tok.encode(" " + answer_text(rel, e), add_special_tokens=False).ids[0]] == 1),
            "single_token_answers": single,
            "most_common_first_tokens": [(dec(i), n) for i, n in firsts.most_common(8)],
            "examples": {str(z): per[z] for z in (1, 6, 8, 17, 26, 29, 47, 79, 92, 118)}}
    # phrasings
    for pid, rels in CANDIDATES.items():
        win = {r: collections.Counter() for r in rels}
        lengths = collections.Counter()
        faults, echo, subj_same, subj_len = [], {r: [] for r in rels}, 0, collections.Counter()
        for e in ELEMENTS:
            c = {r: coords(tok, t, e[2]) for r, t in rels.items()}
            sw = list(range(c["symbol"]["subject"][-1] + 1, c["symbol"]["readout"]))
            nw = list(range(c["number"]["subject"][-1] + 1, c["number"]["readout"]))
            lengths[(len(sw), len(nw))] += 1
            for r, w in (("symbol", sw), ("number", nw)):
                win[r][tuple(dec(c[r]["ids"][p]) for p in w)] += 1
            if [c["symbol"]["ids"][p] for p in c["symbol"]["subject"]] == [c["number"]["ids"][p] for p in c["number"]["subject"]]:
                subj_same += 1
            subj_len[len(c["symbol"]["subject"])] += 1
            for r in rels:
                a = tok.encode(" " + answer_text(r, e), add_special_tokens=False).ids[0]
                sids = [c[r]["ids"][p] for p in c[r]["subject"]]
                bare = tok.encode(answer_text(r, e), add_special_tokens=False).ids[0]
                if a in sids or bare in sids:
                    echo[r].append(e[1])
        rep["phrasings"][pid] = {
            "templates": rels, "window_length_pairs": {f"{a}/{b}": n for (a, b), n in lengths.items()},
            "windows": {r: [[list(k), n] for k, n in v.most_common(3)] for r, v in win.items()},
            "window_ids_constant": {r: len(v) == 1 for r, v in win.items()},
            "equal_window_length_all": len(lengths) == 1 and all(a == b for a, b in lengths),
            "subject_ids_equal_across_relations": subj_same, "subject_token_counts": dict(sorted(subj_len.items())),
            "subject_echo": {r: v for r, v in echo.items()}}
    sp = split_of(0)
    rep["split"] = {"salt": SPLIT_SALT, "fractions": dict(SPLIT),
                    "counts": dict(collections.Counter(sp.values())),
                    "assignment": {str(z): sp[z] for z in sorted(sp)}}
    (out / "feasibility.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps({k: rep[k] for k in ("answers",)}, indent=1, ensure_ascii=False))
    for pid, p in rep["phrasings"].items():
        print(pid, p["window_length_pairs"], p["window_ids_constant"], p["equal_window_length_all"],
              {r: v[0] for r, v in p["windows"].items()}, "subj_eq", p["subject_ids_equal_across_relations"],
              "echo", {r: len(v) for r, v in p["subject_echo"].items()})
    print(rep["split"]["counts"])
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]).resolve()))
