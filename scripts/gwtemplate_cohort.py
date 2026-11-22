"""GW-TEMPLATE-FACTORIAL: the deterministic second-relation cohort rule
(CUR-F, and the same rule for each fallback relation).

Model-free given its inputs. Inputs are fixed facts:

- `classes`: answer class -> the countries whose answer is that class
  (currency name, ISO 4217 code, or one class per country for alpha-3);
- `first_tokens`: answer class -> its accepted first-token IDs;
- `eligible`: the countries eligible in all three phrasings (from the natural
  baseline; before the baseline it can be every country, which gives an
  upper bound).

Algorithm (no manual step):

1. Keep the classes with at least one eligible country.
2. Two classes **conflict** when their first-token sets intersect.
3. Choose a **maximum-cardinality** set of pairwise non-conflicting classes,
   by exact search. Tie-break, in order:
   (a) more eligible countries covered;
   (b) the lexicographically smallest sorted tuple of class names.
4. Every eligible country in an unchosen class goes to the exclusion ledger
   with the reason "answer tokens overlap a chosen class".
5. **Donor rule:** a country needs at least `MIN_DONORS` eligible countries in
   other chosen classes. Countries failing it are removed and ledgered, and
   this repeats until stable. A class left with no country is dropped, and
   step 3 is not re-run: the dropped class is ledgered.
6. **Feasible** only if the resulting cohort has at least `MIN_COUNTRIES`
   countries and at least `MIN_CLASSES` classes, all together.
"""
from __future__ import annotations

MIN_COUNTRIES = 60
MIN_CLASSES = 15
MIN_DONORS = 2
FALLBACK_ORDER = ("currency_name", "currency_code_iso4217", "country_code_iso3166_alpha3")


def _components(nodes, conflicts):
    seen, comps = set(), []
    for n in sorted(nodes):
        if n in seen:
            continue
        stack, comp = [n], []
        seen.add(n)
        while stack:
            x = stack.pop()
            comp.append(x)
            for y in sorted(conflicts[x]):
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        comps.append(sorted(comp))
    return comps


def _best_in_component(comp, conflicts, weight):
    """Exact maximum independent set of one component, with the tie-break
    (size, then covered countries, then lexicographically smallest names)."""
    best = None

    def rec(i, chosen):
        nonlocal best
        if i == len(comp):
            if best is None or _better(chosen, best):
                best = list(chosen)
            return
        c = comp[i]
        if not (conflicts[c] & set(chosen)):
            chosen.append(c)
            rec(i + 1, chosen)
            chosen.pop()
        rec(i + 1, chosen)

    def _better(a, b):
        ka = (len(a), sum(weight[c] for c in a))
        kb = (len(b), sum(weight[c] for c in b))
        if ka != kb:
            return ka > kb
        return tuple(sorted(a)) < tuple(sorted(b))

    if len(comp) > 24:
        raise ValueError(f"a conflict component of {len(comp)} classes is too large for exact search")
    rec(0, [])
    return best


def select(classes: dict, first_tokens: dict, eligible: set) -> dict:
    live = {c: sorted(set(cs) & eligible) for c, cs in classes.items() if set(cs) & eligible}
    conflicts = {c: {d for d in live if d != c and set(first_tokens[c]) & set(first_tokens[d])} for c in live}
    weight = {c: len(v) for c, v in live.items()}
    chosen = []
    for comp in _components(live, conflicts):
        chosen += _best_in_component(comp, conflicts, weight)
    chosen = sorted(chosen)
    ledger = [{"country": k, "class": c, "reason": "answer tokens overlap a chosen class"}
              for c in sorted(set(live) - set(chosen)) for k in live[c]]
    members = {c: list(live[c]) for c in chosen}
    while True:
        total = sum(len(v) for v in members.values())
        drop = [(c, k) for c, v in members.items() for k in v if total - len(v) < MIN_DONORS]
        if not drop:
            break
        for c, k in drop:
            members[c].remove(k)
            ledger.append({"country": k, "class": c, "reason": f"fewer than {MIN_DONORS} donors in other classes"})
    for c in [c for c, v in members.items() if not v]:
        ledger.append({"country": None, "class": c, "reason": "class left empty by the donor rule"})
        del members[c]
    countries = sorted(k for v in members.values() for k in v)
    return {"selected_classes": chosen, "classes": sorted(members), "members": members, "countries": countries,
            "n_countries": len(countries), "n_classes": len(members),
            "feasible": len(countries) >= MIN_COUNTRIES and len(members) >= MIN_CLASSES,
            "ledger": ledger, "conflict_edges": sum(len(v) for v in conflicts.values()) // 2}


def decide(results_in_order: list[tuple[str, dict]]) -> dict:
    """The first relation in FALLBACK_ORDER whose cohort is feasible, else STOP."""
    names = [n for n, _ in results_in_order]
    if names != list(FALLBACK_ORDER[:len(names)]):
        raise ValueError("relations must be evaluated in the frozen fallback order")
    for name, res in results_in_order:
        if res["feasible"]:
            return {"relation": name, "cohort": res}
    return {"relation": "STOP", "reason": "no relation meets the frozen feasibility rule"}
