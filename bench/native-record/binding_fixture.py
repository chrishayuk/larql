"""Answer-free reader splits and a separate frozen factorial write bank."""
from run_address import DECOYS

RELATIONS = ['capital', 'currency', 'language']
ROSTERS = {'development': ['Merovia', 'Tarskeld', 'Pelvoria', 'Nuskara'],
           'evaluation': ['Quenvar', 'Drelmora', 'Surneth', 'Velkora'],
           'factorial': ['Avenlorn', 'Braskovia', 'Celdrune', 'Dornessa']}
TRAIN = ["{e}'s {r} is", "For {e}, the {r} is"]
VALID = ['In {e}, the {r} is', 'The {r} associated with {e} is']
FRESH = ['With regard to {e}, its {r} is',
         'If one asks about the {r} of {e}, the answer is',
         "Looking up {e}'s {r}, one finds", 'The recorded {r} for {e} is']
ALIASES = ['{a} denotes {e} in this register. The {r} of {a} is',
           'The place called {a} has the full name {e}. Its {r} is']


def bindings(roster):
    return [(e, r) for e in ROSTERS[roster] for r in RELATIONS]


def canonical(e, r):
    return f'The {r} of {e} is'


def reader_fixture(roster):
    result = []

    def add(group, prompt, label, relation):
        result.append(dict(id=f'{roster}-{group}-{len(result):03}', group=group,
                           prompt=prompt, label=label, relation=relation))

    for label, (e, r) in enumerate(bindings(roster)):
        add('enroll', canonical(e, r), label, r)
        if roster == 'development':
            for form in TRAIN:
                add('train', form.format(e=e, r=r), label, r)
            for form in VALID:
                add('validation', form.format(e=e, r=r), label, r)
        else:
            for form in FRESH:
                add('test', form.format(e=e, r=r), label, r)
            for form in ALIASES:
                add('alias', form.format(e=e, r=r, a=e[:3] + '-district'), label, r)
        # Clearly different spellings with no alias declaration, held out of enrollment.
        near = e[:-1] + ('ix' if roster == 'development' else 'eth')
        form = VALID[0] if roster == 'development' else FRESH[0]
        add('unknown', form.format(e=near, r=r), None, r)
    assert len(set(row['prompt'] for row in result)) == len(result)
    return result


def factorial_fixture():
    values = [['Oslo', 'Paris', 'Rome', 'Tokyo'], ['Yen', 'Euro', 'Dollar', 'Pound'],
              ['Welsh', 'French', 'German', 'Spanish']]
    facts = [(e, r, values[j][(i + j) % 4]) for i, e in enumerate(ROSTERS['factorial'])
             for j, r in enumerate(RELATIONS)]
    rows = []

    def add(group, prompt, fact=None):
        rows.append(dict(id=f'{group}-{len(rows):03}', group=group,
                         prompt=prompt, fact=fact, expected=facts[fact][2] if fact is not None else None))

    for i, (e, r, _) in enumerate(facts):
        add('installation', canonical(e, r), i)
        for form in FRESH:
            add('sentence', form.format(e=e, r=r), i)
        for form in ALIASES:
            add('alias', form.format(e=e, r=r, a=e[:3] + '-district'), i)
        add('similar_name', canonical(e[:-1] + 'eth', r))
    for e in ['Estonia', 'Latvia', 'Lithuania', 'Iceland', 'Romania', 'Bulgaria',
              'Croatia', 'Slovenia', 'Slovakia', 'Serbia', 'Albania', 'Malta',
              'Luxembourg', 'Switzerland', 'Netherlands', 'Belgium', 'Nepal',
              'Bhutan', 'Mongolia', 'Indonesia']:
        for r in RELATIONS:
            add('other_entity', canonical(e, r))
    for prompt in ['A compass needle points toward', 'The freezing point of water is',
                   'A spider has eight', 'The Earth orbits the', 'Plants absorb carbon',
                   'The number after nineteen is', 'The main ingredient in bread is',
                   'A violin is played with a', 'The organ that pumps blood is',
                   'The past tense of go is', 'A baker works in a', 'Winter is followed by']:
        add('unrelated', prompt)
    assert len({row['prompt'] for row in rows}) == len(rows)
    assert set(DECOYS).isdisjoint(row['prompt'] for row in rows)
    for row in rows:
        assert all(v.casefold() not in row['prompt'].casefold() for group in values for v in group)
    return facts, rows
