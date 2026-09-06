"""Frozen development partitions and fresh native-controller evaluation banks."""
from binding_fixture import factorial_fixture, canonical, FRESH, ALIASES, RELATIONS

EVAL_FORMS = [
    'According to the register, the {r} of {e} is',
    '{e} has its {r} listed as',
    'The entry for {e} gives its {r} as',
    'In the country register, {e} has a {r} that is',
]
EVAL_ALIASES = [
    'This register uses {a} as another name for {e}. The {r} of {a} is',
    '{e}, also referred to here as {a}, has its {r} listed as',
]
NEW_ENTITIES = ['Eldavren', 'Feskoria', 'Galdreth', 'Hurnovia']
NEW_OTHER = ['Jandovia', 'Kelbrune', 'Lestaria', 'Mardovin', 'Neldrava', 'Orskelia',
             'Paldrune', 'Raskoven', 'Selmoria', 'Teldrava', 'Uskelorn', 'Vardelia']
UNRELATED = ['A triangle has three', 'Birds build their nests in', 'A thermometer measures',
             'The opposite of empty is', 'Rain falls from', 'People use umbrellas when it',
             'A week contains seven', 'The plural of tooth is', 'The color of fresh snow is',
             'A carpenter works with', 'The ocean contains salt', 'A clock measures']


def fixtures():
    old_facts, old_rows = factorial_fixture()
    new_facts = [(NEW_ENTITIES[i // 3], r, v) for i, (_, r, v) in enumerate(old_facts)]
    result = {}
    for bank, facts in [('observed_roster', old_facts), ('fresh_roster', new_facts)]:
        development = []
        negative_index = 0
        for original in old_rows:
            row = dict(original)
            if bank == 'fresh_roster':
                for old, new in zip([f[0] for f in old_facts[::3]], NEW_ENTITIES):
                    row['prompt'] = row['prompt'].replace(old[:-1] + 'eth', new[:-1] + 'eth')
                    row['prompt'] = row['prompt'].replace(old[:3] + '-district', new[:3] + '-district')
                    row['prompt'] = row['prompt'].replace(old, new)
            if row['fact'] is not None:
                i = row['fact']
                old_e, old_r, _ = old_facts[i]
                held = [FRESH[-1].format(e=old_e, r=old_r),
                        ALIASES[-1].format(e=old_e, r=old_r, a=old_e[:3] + '-district')]
                row['split'] = 'validation' if original['prompt'] in held else 'train'
            else:
                row['split'] = 'validation' if negative_index % 3 == 0 else 'train'
                negative_index += 1
            development.append(row)
        evaluation = []

        def add(group, prompt, fact=None):
            evaluation.append(dict(id=f'{group}-{len(evaluation):03}', group=group, prompt=prompt,
                fact=fact, expected=facts[fact][2] if fact is not None else None))

        aliases = ['Northhaven', 'Eastmere', 'Southwick', 'Westford']
        for i, (e, r, _) in enumerate(facts):
            add('installation', canonical(e, r), i)
            for form in EVAL_FORMS:
                add('sentence', form.format(e=e, r=r), i)
            for form in EVAL_ALIASES:
                add('alias', form.format(e=e, r=r, a=aliases[i // 3]), i)
            add('similar_name', canonical(e + 'a', r))
            add('similar_name', EVAL_FORMS[0].format(e=e[:-2] + 'un', r=r))
        for e in NEW_OTHER:
            for r in RELATIONS:
                add('other_entity', EVAL_FORMS[1].format(e=e, r=r))
        for e, _, _ in facts[::3]:
            for r in ['continent', 'population', 'anthem']:
                add('other_relation', EVAL_FORMS[0].format(e=e, r=r))
        for prompt in UNRELATED:
            add('unrelated', prompt)
        assert len(development) == len(evaluation) == 168
        assert len({r['prompt'] for r in evaluation}) == 168
        observed = {r['prompt'] for r in development}
        assert all(r['prompt'] not in observed for r in evaluation if r['group'] != 'installation')
        result[bank] = dict(facts=facts, development=development, evaluation=evaluation,
                            replacement=dict(fact=0, value='Berlin'))
    return result
