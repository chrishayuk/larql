#!/usr/bin/env python3
"""Generate the deliberately boring MAD-V3 graph-lookup corpus."""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer


RELATIONS = (
    (
        "facility",
        (
            "What facility is associated with {source}?",
            "Give me {source}'s facility.",
            "{source} maps to which facility?",
            "Following the facility field for {source}, what value is stored?",
            "Resolve {source} through its facility attribute.",
        ),
    ),
    (
        "supervisor",
        (
            "Who supervises {source}?",
            "Give me {source}'s supervisor.",
            "{source} reports to which supervisor?",
            "Following the supervisor field for {source}, what value is stored?",
            "Resolve {source} through its supervisor attribute.",
        ),
    ),
    (
        "route",
        (
            "What route leaves {source}?",
            "Give me {source}'s route.",
            "{source} maps to which route?",
            "Following the route field for {source}, what value is stored?",
            "Resolve {source} through its route attribute.",
        ),
    ),
    (
        "archive",
        (
            "What archive is associated with {source}?",
            "Give me {source}'s archive.",
            "{source} maps to which archive?",
            "Following the archive field for {source}, what value is stored?",
            "Resolve {source} through its archive attribute.",
        ),
    ),
)

# Adversarial profile: syntax is crossed with operation, entity-name families
# are permuted independently per graph, and the hard C arm uses an operation
# synonym that never appears in history. The record schema keeps the canonical
# relation label, so the model must connect the held-out query term to the same
# database operation rather than match a shared token.
ADVERSARIAL_ALIASES = {
    "facility": ("facility", "site", "premises", "venue"),
    "supervisor": ("supervisor", "manager", "overseer", "director"),
    "route": ("route", "path", "course", "way"),
    "archive": ("archive", "repository", "collection", "storehouse"),
}
ADVERSARIAL_TEMPLATES = (
    "Which {field} value belongs to {source}?",
    "Give the {field} entry for {source}.",
    "Look up {source} under {field}.",
    "For {source}, return the value in the {field} column.",
    "Resolve the {field} associated with {source}.",
    "Read {source}'s {field} value.",
    "Under {field}, which entry is recorded for {source}?",
    "Find {source}, then return its {field}.",
    "What does the {field} record say for {source}?",
    "From the {source} row, extract {field}.",
    "Return the value reached through {field} from {source}.",
    "Use {field} to resolve {source}.",
)

CURATED_ANSWER_WORDS = """
Aaron Abigail Adam Adrian Aiden Alan Albert Alex Alice Amanda Amber Amelia Amy
Andrea Andrew Angela Anna Anthony Arthur Ashley Austin Ava Barbara Benjamin
Bethany Blake Brandon Brenda Brian Brittany Brooke Bruce Caleb Cameron Carl
Carol Caroline Catherine Charles Charlotte Chloe Christina Christopher Claire
Clara Colin Connor Daniel Danielle David Deborah Dennis Diana Dominic Donna
Dorothy Dylan Edward Eleanor Elijah Elizabeth Ella Emily Emma Eric Ethan Eva
Evelyn Fiona Florence Frances Frank Gabriel George Georgia Grace Graham Hannah
Harold Harry Hazel Heather Helen Henry Holly Isaac Isabella Jack Jacob James
Jane Jasmine Jason Jennifer Jessica Joan John Jonathan Jordan Joseph Joshua
Julia Julian Justin Karen Katherine Kathleen Katie Kevin Kimberly Laura Lauren
Leah Leo Leonard Liam Lily Linda Lisa Logan Louis Lucy Luke Madison Maria Mark
Martin Mary Mason Matthew Megan Melissa Mia Michael Michelle Molly Nancy Natalie
Nathan Nicholas Nicole Noah Oliver Olivia Oscar Owen Pamela Patricia Paul Peter
Philip Rachel Rebecca Richard Riley Robert Rose Ruby Russell Ryan Samantha Samuel
Sandra Sarah Scott Sean Sebastian Simon Sophia Sophie Stephen Steven Susan
Taylor Teresa Thomas Timothy Toby Tracy Tyler Valerie Victoria Vincent Walter
William Willow Zachary London Paris Berlin Madrid Lisbon Vienna Dublin Prague
Warsaw Athens Oslo Rome Milan Venice Tokyo Kyoto Seoul Sydney Perth Boston Denver
Dallas Miami Seattle Phoenix Chicago Detroit Toronto Ottawa Cairo Nairobi Lima
Quito Havana Bristol Oxford Cambridge York Durham Exeter Brighton Cardiff
Glasgow Dundee Aberdeen Brussels Zurich Geneva Munich Hamburg Cologne Naples
Florence Turin Valencia Seville Granada Porto Rotterdam Utrecht Antwerp
Stockholm Helsinki Tallinn Riga Vilnius Sofia Zagreb Belgrade Sarajevo Skopje
Tirana Ankara Izmir Beirut Amman Tunis Rabat Accra Lagos Dakar Kigali Kampala
Harare Lusaka Jaipur Mumbai Delhi Chennai Kolkata Bangkok Manila Jakarta Hanoi
Osaka Sapporo Taipei Auckland Wellington Darwin Adelaide Houston Orlando
Atlanta Memphis Nashville Portland Tacoma Oakland Tucson Fresno Raleigh
Anderson Bailey Barnes Bell Bennett Brooks Butler Campbell Carter Chapman
Clark Collins Cook Cooper Cox Crawford Cunningham Dawson Dixon Douglas Dunn
Edwards Elliott Ellis Evans Fisher Fleming Ford Foster Fowler Fox Franklin
Freeman Gardner Gibson Gilbert Gordon Grant Gray Green Griffin Hall Hamilton
Hansen Harper Harris Harrison Harvey Henderson Hicks Holmes Howard Hudson Hunt
Hunter Jackson Jacobs Jenkins Jennings Johnston Jones Kelley Kennedy Kim King
Knight Lane Larson Lawrence Lawson Lee Lewis Lloyd Long Lowe Marshall Martin
Mason Matthews Maxwell Meyer Mills Mitchell Montgomery Moore Morgan Morris
Murray Myers Newman Nichols Norris Olson Palmer Parker Patterson Pearson Perry
Peters Pierce Porter Powell Price Ramsey Reed Reeves Reid Reynolds Rhodes Rice
Richards Richardson Riley Roberts Robertson Rogers Rose Ross Russell Sanders
Schmidt Shaw Simpson Snyder Spencer Stevens Stewart Stone Sullivan Sutton Tate
Terry Thornton Townsend Tucker Turner Wagner Walker Wallace Warren Waters
Watkins Watson Watts Weaver Webb Wells Wheeler White Williams Willis Wilson
Wood Wright Young
""".split()

PREFIXES = (
    "Rook",
    "Vant",
    "North",
    "Ember",
    "Calder",
    "Thorn",
    "Mere",
    "Lark",
    "Dun",
    "Kest",
    "Orin",
    "Pell",
    "Quill",
    "Sable",
    "Tarn",
    "Wren",
)
SUFFIXES = (
    "haven",
    "rel",
    "bridge",
    "mere",
    "wick",
    "ford",
    "spire",
    "vale",
    "crest",
    "mark",
    "gate",
    "fall",
    "reach",
    "stead",
    "ward",
    "holt",
)


def name(index: int) -> str:
    """Return a unique synthetic entity without a natural-language type hint."""
    prefix = PREFIXES[(index * 7 + 3) % len(PREFIXES)]
    suffix = SUFFIXES[(index * 11 + 5) % len(SUFFIXES)]
    return f"{prefix}{suffix}-{index:04d}"


def layout_permutation(layout_seed: int, size: int) -> list[int]:
    order = list(range(size))
    random.Random(0x4D414456334C4159 + layout_seed).shuffle(order)
    return order


def graph_facts(graph: int, profile: str) -> list[tuple[str, str, str]]:
    facts = []
    # The pilot's stride aliases each relation onto one recurring name family.
    # An odd adversarial stride walks every prefix/suffix residue while keeping
    # each graph's eight entity indices disjoint.
    base = graph * (9 if profile == "adversarial" else 16)
    entity_slots = list(range(len(RELATIONS)))
    if profile == "adversarial":
        random.Random(0x4D41445633000000 + graph).shuffle(entity_slots)
    for relation_index, (relation, _templates) in enumerate(RELATIONS):
        entity_slot = entity_slots[relation_index]
        source = name(base + entity_slot * 2)
        target = name(base + entity_slot * 2 + 1)
        facts.append((relation, source, target))
    return facts


def alias_for_template(relation: str, template_index: int, alias_fold: int) -> tuple[int, str]:
    seen = [index for index in range(4) if index != alias_fold]
    if template_index < 3:
        alias_index = seen[template_index]
    elif template_index == 3:
        alias_index = seen[0]
    else:
        alias_index = alias_fold
    return alias_index, ADVERSARIAL_ALIASES[relation][alias_index]


def prompt_for(
    facts,
    relation,
    source,
    template_index,
    filler_index,
    profile,
    alias_fold=3,
    alias_index=None,
    layout_seed=None,
) -> str:
    records = [f"{src} | {rel} | {dst}" for rel, src, dst in facts]
    if profile == "adversarial":
        if layout_seed is None:
            layout_seed = filler_index
        order = layout_permutation(layout_seed, len(records))
        records = [records[index] for index in order]
        if alias_index is None:
            alias_index, field = alias_for_template(relation, template_index, alias_fold)
        else:
            field = ADVERSARIAL_ALIASES[relation][alias_index]
        query = ADVERSARIAL_TEMPLATES[template_index % len(ADVERSARIAL_TEMPLATES)].format(
            field=field, source=source
        )
    else:
        if filler_index % 3 == 1:
            records = list(reversed(records))
        elif filler_index % 3 == 2:
            records = records[2:] + records[:2]
        template = dict(RELATIONS)[relation][template_index]
        query = template.format(source=source)
    filler = (
        "Return only the stored value.",
        "Use the records below; unrelated fields are distractors.",
        "Treat this as a database lookup and answer from the records.",
    )[filler_index % 3]
    return (
        "Database records:\n"
        + "\n".join(records)
        + f"\n{filler}\nQuery: {query}\nAnswer:"
    )


def row(
    graph,
    relation_index,
    template_index,
    filler_index,
    split,
    regime,
    profile,
    alias_fold=3,
    alias_index=None,
    variant=None,
):
    relation, _templates = RELATIONS[relation_index]
    facts = graph_facts(graph, profile)
    _relation, source, target = facts[relation_index]
    if profile == "adversarial":
        if alias_index is None:
            resolved_alias_index, _field = alias_for_template(
                relation, template_index, alias_fold
            )
        else:
            resolved_alias_index = alias_index
        layout_seed = graph * 257 + relation_index * 31 + filler_index
    else:
        resolved_alias_index = None
        layout_seed = filler_index
    variant_suffix = "" if variant is None else f"-v{variant:03d}"
    result = {
        "id": f"g{graph:04d}-{relation}-t{template_index}-f{filler_index}{variant_suffix}-{regime}",
        "semantic_id": f"graph:{graph:04d}:lookup:{relation}:source:{source}",
        "operation_id": f"lookup:{relation}",
        "wording_id": (
            f"{relation}:template:{template_index}"
            if profile == "pilot"
            else f"template:{template_index}:alias:{resolved_alias_index}"
        ),
        "split": split,
        "group_id": f"graph:{graph:04d}",
        "regime": regime,
        "prompt": prompt_for(
            facts,
            relation,
            source,
            template_index,
            filler_index,
            profile,
            alias_fold,
            resolved_alias_index,
            layout_seed,
        ),
        "expected": target,
    }
    if resolved_alias_index is not None:
        result["alias_id"] = f"{relation}:{ADVERSARIAL_ALIASES[relation][resolved_alias_index]}"
        result["layout_id"] = "order:" + "-".join(
            str(index) for index in layout_permutation(layout_seed, len(facts))
        )
    return result


def build(train_graphs: int, heldout_graphs: int, profile: str, alias_fold: int) -> list[dict]:
    rows = []
    for graph in range(train_graphs):
        for relation_index in range(len(RELATIONS)):
            for template_index in range(3):
                rows.append(
                    row(
                        graph,
                        relation_index,
                        template_index,
                        template_index,
                        "history",
                        "history",
                        profile,
                        alias_fold,
                    )
                )
            rows.append(
                row(
                    graph,
                    relation_index,
                    3,
                    graph + relation_index,
                    "query",
                    "a_same_graph_unseen_wording",
                    profile,
                    alias_fold,
                )
            )
    for graph in range(train_graphs, train_graphs + heldout_graphs):
        for relation_index in range(len(RELATIONS)):
            rows.append(
                row(
                    graph,
                    relation_index,
                    0,
                    graph + relation_index,
                    "query",
                    "b_unseen_graph_seen_wording",
                    profile,
                    alias_fold,
                )
            )
            rows.append(
                row(
                    graph,
                    relation_index,
                    4,
                    graph + relation_index + 1,
                    "query",
                    "c_unseen_graph_unseen_wording",
                    profile,
                    alias_fold,
                )
            )
    return rows


def build_route(
    train_graphs: int, heldout_graphs: int, route_variants: int, alias_fold: int
) -> list[dict]:
    """Repeated executions per answer for residual-only basin discovery.

    History and query graphs remain disjoint. Each operation/answer cell has
    multiple syntax, alias and record-layout realisations, which makes it
    possible to reject communities that merely encode answer identity.
    """
    rows = []
    seen_aliases = [index for index in range(4) if index != alias_fold]
    for graph in range(train_graphs):
        for relation_index in range(len(RELATIONS)):
            for variant in range(route_variants):
                rows.append(
                    row(
                        graph,
                        relation_index,
                        4 + variant % (len(ADVERSARIAL_TEMPLATES) - 4),
                        variant * 7 + graph,
                        "history",
                        "route_history",
                        "adversarial",
                        alias_fold,
                        seen_aliases[variant % len(seen_aliases)],
                        variant,
                    )
                )
    for graph in range(train_graphs, train_graphs + heldout_graphs):
        for relation_index in range(len(RELATIONS)):
            for variant in range(route_variants):
                rows.append(
                    row(
                        graph,
                        relation_index,
                        4 + variant % (len(ADVERSARIAL_TEMPLATES) - 4),
                        variant * 11 + graph,
                        "query",
                        "route_heldout",
                        "adversarial",
                        alias_fold,
                        alias_fold,
                        variant,
                    )
                )
    return rows


def use_single_token_answers(rows: list[dict], tokenizer) -> None:
    """Replace synthetic multi-token targets with neutral one-token values."""
    targets = sorted({item["expected"] for item in rows})
    blocked = {
        word.lower()
        for aliases in ADVERSARIAL_ALIASES.values()
        for word in aliases
    } | {
        "answer",
        "database",
        "entry",
        "field",
        "query",
        "record",
        "return",
        "value",
    }
    candidates = []
    seen_tokens = set()
    for word in CURATED_ANSWER_WORDS:
        encoded = tokenizer(" " + word, add_special_tokens=False)["input_ids"]
        if len(encoded) != 1 or encoded[0] in seen_tokens or word.lower() in blocked:
            continue
        decoded = tokenizer.decode(
            encoded, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        if decoded != " " + word:
            continue
        seen_tokens.add(encoded[0])
        candidates.append((int(encoded[0]), word))
    if len(candidates) < len(targets):
        raise ValueError(
            f"tokenizer exposes only {len(candidates)} eligible one-token answers; "
            f"need {len(targets)}"
        )
    random.Random(0x524F55544531414E).shuffle(candidates)
    replacement = {
        target: candidates[index][1] for index, target in enumerate(targets)
    }
    for item in rows:
        old = item["expected"]
        new = replacement[old]
        if item["prompt"].count(old) != 1:
            raise ValueError(f"target {old} is not unique in prompt {item['id']}")
        item["prompt"] = item["prompt"].replace(old, new)
        item["expected"] = new


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="local Hugging Face model/tokenizer directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-graphs", type=int, default=300)
    parser.add_argument("--heldout-graphs", type=int, default=100)
    parser.add_argument(
        "--profile",
        choices=("pilot", "adversarial"),
        default="adversarial",
        help="pilot reproduces P1; adversarial breaks lexical/entity/layout coupling",
    )
    parser.add_argument(
        "--alias-fold",
        type=int,
        choices=range(4),
        default=3,
        help="alias index held out from history and used by held-out queries",
    )
    parser.add_argument(
        "--route-variants",
        type=int,
        default=0,
        help="emit ROUTE-1 repeated cells instead of the A/B/C corpus",
    )
    parser.add_argument(
        "--route-answer-tokens",
        choices=("one", "synthetic"),
        default="one",
        help="one uses unique tokenizer-verified one-token answer identities",
    )
    args = parser.parse_args()

    if args.route_variants < 0:
        parser.error("--route-variants must be non-negative")
    if args.route_variants and args.profile != "adversarial":
        parser.error("--route-variants requires --profile adversarial")
    if args.route_variants:
        rows = build_route(
            args.train_graphs,
            args.heldout_graphs,
            args.route_variants,
            args.alias_fold,
        )
    else:
        rows = build(args.train_graphs, args.heldout_graphs, args.profile, args.alias_fold)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.route_variants and args.route_answer_tokens == "one":
        use_single_token_answers(rows, tokenizer)
    lengths = []
    for item in rows:
        prompt_ids = [int(token) for token in tokenizer(item["prompt"])["input_ids"]]
        expected_token_ids = [
            int(token)
            for token in tokenizer(
                " " + item["expected"], add_special_tokens=False
            )["input_ids"]
        ]
        if not expected_token_ids:
            raise ValueError(f"answer continuation has no tokens for {item['id']}")
        item["token_ids"] = prompt_ids
        item["expected_token_ids"] = expected_token_ids
        lengths.append(len(item["token_ids"]))

    rng = random.Random(0x4D414456334B4E4E)
    rng.shuffle(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for item in rows:
            handle.write(json.dumps(item, separators=(",", ":")) + "\n")

    regimes = Counter(item["regime"] for item in rows)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "profile": args.profile,
                "alias_fold": args.alias_fold,
                "route_variants": args.route_variants,
                "route_answer_tokens": args.route_answer_tokens,
                "regimes": regimes,
                "tokens_total": sum(lengths),
                "tokens_min": min(lengths),
                "tokens_mean": sum(lengths) / len(lengths),
                "tokens_max": max(lengths),
                "output": str(args.out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
