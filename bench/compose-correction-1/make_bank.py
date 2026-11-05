#!/usr/bin/env python3
"""Deterministic generator for the COMPOSE-CORRECTION-1 prompt bank.

Writes bank.json next to this file and prints its SHA-256. The registered bank is the
output of this script; the hash is recorded at freeze. Nothing here reads a model.

Prompts seen during the exploratory investigation are excluded by construction:
Atlantis, France, Germany, and "The weather today is".
"""
import hashlib, json, pathlib, sys

SEEN = {"Atlantis", "France", "Germany", "The weather today is"}

# 4 relations x 3 invented entities; targets are single words, first-token scored.
# canonical prompt = "The {relation} of {entity} is" (relations are bare nouns so it is well formed).
FACTS = [
    ("capital", "Zorvania", "Meridian"), ("capital", "Eldoria", "Obsidian"), ("capital", "Kelmar", "Solstice"),
    ("ruler", "Zorvania", "Aldric"), ("ruler", "Eldoria", "Brannoch"), ("ruler", "Kelmar", "Corvane"),
    ("currency", "Vossmere", "Lumen"), ("currency", "Thandor", "Verdan"), ("currency", "Ilmar", "Marrow"),
    ("language", "Vossmere", "Orrish"), ("language", "Thandor", "Tavrin"), ("language", "Ilmar", "Belmaric"),
]
# Used only if a registered fact's baseline target first-token probability is >= 0.01 (rule in the
# pre-registration); taken in order, never chosen after seeing results.
RESERVE = [
    ("capital", "Quillon", "Cinder"), ("ruler", "Quillon", "Dravick"),
    ("currency", "Quillon", "Ember"), ("language", "Quillon", "Ashkari"),
]

NEIGHBOUR_ENTITIES = {
    "capital": ["Italy", "Spain", "Japan", "Canada", "Egypt", "Brazil", "Poland", "Norway"],
    "ruler": ["Monaco", "Egypt", "Japan", "Thailand", "Jordan", "Morocco", "Sweden", "Belgium"],
    "currency": ["Japan", "Brazil", "India", "Mexico", "Sweden", "Poland", "Switzerland", "Turkey"],
    "language": ["Brazil", "Egypt", "Japan", "Mexico", "Iran", "Poland", "Thailand", "Hungary"],
}
# Different template, related subject (countries and what is said about them).
RELATED = {
    "capital": ["The largest city in Italy is", "Spain is a country in", "The river that runs through Egypt is the",
                "Japan is an island nation in", "The national animal of Canada is the", "Brazil is the largest country in",
                "The official name of Poland is", "Norway is known for its"],
    "ruler": ["The head of state of Japan is the", "A monarchy is a country ruled by a", "The title of the ruler of Monaco is",
              "Sweden is a constitutional", "The king of Belgium lives in", "A republic is governed by a",
              "The ruler of ancient Egypt was called the", "Thailand is a kingdom in"],
    "currency": ["The coin worth one hundred cents is", "Japan buys and sells goods using", "A bank holds your",
                 "The price of a loaf of bread in Mexico is paid in", "Exchange rates compare one", "India is a country in",
                 "Switzerland is famous for its", "Turkey is a country in"],
    "language": ["People in Mexico speak", "A dialect is a form of", "The alphabet used in Poland is the",
                 "Hungary is a country in", "Thailand is a country in", "Egypt is a country in",
                 "The script used to write Japanese includes", "Brazil is a country in"],
}

facts_out = []
for rel, ent, tgt in FACTS:
    facts_out.append({
        "relation": rel, "entity": ent, "target": tgt,
        "canonical_prompt": f"The {rel} of {ent} is",
        "same_template": [f"The {rel} of {e} is" for e in NEIGHBOUR_ENTITIES[rel]],
        "related": RELATED[rel],
    })
reserve_out = [{"relation": r, "entity": e, "target": t, "canonical_prompt": f"The {r} of {e} is",
                "same_template": [f"The {r} of {x} is" for x in NEIGHBOUR_ENTITIES[r]], "related": RELATED[r]}
               for r, e, t in RESERVE]

unrelated = []
unrelated += [  # factual completions, no countries or capitals
    "Water boils at", "The chemical symbol for gold is", "The largest planet in the solar system is", "A triangle has",
    "The speed of light is approximately", "The human heart has", "Photosynthesis converts sunlight into",
    "The freezing point of water is", "The Pacific is the largest", "A year on Earth takes",
    "The chemical formula for table salt is", "Sound travels fastest through", "The smallest prime number is",
    "An octopus has", "The tallest mountain on Earth is", "DNA stands for", "The moon orbits the",
    "A leap year has", "The atomic number of carbon is", "Iron rusts when it reacts with",
    "The boiling point of nitrogen is", "A group of wolves is called a", "The first element in the periodic table is",
    "Bees collect", "The longest bone in the human body is the", "Lightning is caused by", "A kilobyte is",
    "The currency unit worth one-hundredth of a dollar is the", "Honey is made by", "The Sahara is a",
]
unrelated += [f"{a} {op} {b} =" for a, op, b in [
    (2, "+", 2), (7, "*", 8), (15, "-", 9), (81, "/", 9), (12, "+", 30), (6, "*", 7), (100, "-", 1), (3, "*", 3),
    (45, "+", 55), (9, "*", 9), (64, "/", 8), (250, "-", 50), (11, "*", 11), (19, "+", 23), (72, "/", 12),
    (5, "*", 12), (1000, "-", 1), (33, "+", 67), (8, "*", 8), (144, "/", 12)]]
unrelated += [
    "def add(a, b):\n    return", "for i in range(10):\n    print(", "import numpy as", "SELECT name FROM users WHERE",
    "<html>\n<body>\n<h1>", "x = [1, 2, 3]\nprint(len(", "while True:\n    if", "class Dog:\n    def __init__(self",
    "#include <stdio.h>\nint main() {", "fn main() {\n    println!(", "function square(n) {\n  return",
    "git commit -m", "const items = [1, 2, 3].map(", "if x > 0:\n    y =", "ls -la /", "try:\n    f = open(",
    "{\"name\": \"Ada\", \"age\":", "public static void main(String[] args) {", "print(\"Hello,", "a, b = b, a +",
]
unrelated += [
    "Once upon a time", "The quick brown fox", "To be or not to be", "In the beginning", "It was a dark and stormy",
    "Dear Sir or Madam,", "My favourite hobby is", "The best way to learn is", "Last summer we", "She opened the door and",
    "There are three things I", "Yesterday I saw", "The old man sat", "In my opinion,", "Breaking news:",
    "Chapter 1\n\n", "The meeting will start at", "Thank you for", "I would like to", "After the storm,",
    "Every morning, the baker", "If I could travel anywhere,", "The recipe calls for", "At the end of the day,",
    "Long ago in a forest,", "The train to the city", "He never expected that", "Welcome to", "Under the bridge,", "On Sunday afternoons,",
]
unrelated += [
    "Q: What is the largest ocean?\nA:", "Q: How many days are in a week?\nA:", "Q: What colour is the sky on a clear day?\nA:",
    "Q: What do cows drink?\nA:", "Instruction: Name a fruit.\nResponse:", "Instruction: Say hello in a friendly way.\nResponse:",
    "User: Hi there!\nAssistant:", "User: Can you help me with my homework?\nAssistant:", "Q: Who wrote Romeo and Juliet?\nA:",
    "Q: What is 10 divided by 2?\nA:", "Instruction: Give an example of a mammal.\nResponse:", "User: What time is it?\nAssistant:",
    "Q: What is the opposite of hot?\nA:", "Instruction: Spell the word cat.\nResponse:", "Q: Which animal barks?\nA:",
    "User: Tell me a joke.\nAssistant:", "Q: How many legs does a spider have?\nA:", "Instruction: Name a primary colour.\nResponse:",
    "Q: What season comes after winter?\nA:", "User: Thanks!\nAssistant:",
]
unrelated += [
    'The French word for "cat" is', 'The Spanish word for "water" is', 'The German word for "house" is', 'The Italian word for "thank you" is',
    '"Good morning" in Spanish is', '"Thank you" in French is', 'The Latin word for "love" is', 'The Japanese word for "tree" is',
    '"Hello" in German is', 'The French word for "bread" is', '"Goodbye" in Italian is', 'The Spanish word for "dog" is',
    'The German word for "book" is', '"Please" in French is', 'The Portuguese word for "friend" is', '"Yes" in Japanese is',
    'The Dutch word for "bicycle" is', '"Good night" in Spanish is', 'The Russian word for "peace" is', '"Welcome" in Italian is',
]
unrelated += [
    "Monday, Tuesday, Wednesday,", "January, February, March,", "red, orange, yellow,", "one, two, three, four,", "a, b, c, d,",
    "apples, bananas, cherries,", "north, south, east,", "spring, summer, autumn,", "dog, cat, bird,", "first, second, third,",
    "cheese, bread, butter,", "Mercury, Venus, Earth,", "circle, square, triangle,", "morning, afternoon, evening,", "2, 4, 6, 8,",
    "hydrogen, helium, lithium,", "violin, cello, flute,", "salt, pepper, sugar,", "Alpha, Beta, Gamma,", "hat, scarf, gloves,",
]
unrelated += [
    "A noun is a word that", "Gravity is the force that", "An algorithm is", "A metaphor is", "Democracy is a system in which",
    "A mammal is an animal that", "Inflation is", "A verb is a word that", "Evaporation is the process by which",
    "A sonnet is a poem of", "An atom is the smallest unit of", "A budget is a plan for", "A predator is an animal that",
    "Friction is the force that", "A synonym is a word that", "An ecosystem is", "A haiku is a poem with",
    "A fraction represents", "An adjective describes", "Momentum is the product of",
]

unrelated += [
    "A stitch in time saves", "The early bird catches the", "Actions speak louder than", "Don't count your chickens before they",
    "A penny saved is a penny", "All that glitters is not", "Better late than", "Birds of a feather flock",
    "Every cloud has a silver", "Look before you", "Many hands make light", "The pen is mightier than the",
    "When in Rome, do as the Romans", "You can't judge a book by its", "Practice makes", "Too many cooks spoil the",
    "Two wrongs don't make a", "An apple a day keeps the doctor", "Out of sight, out of", "Rome wasn't built in a",
]

assert len(unrelated) == 200
assert len(set(unrelated)) == 200
for p in unrelated:
    assert p not in SEEN
for f in facts_out + reserve_out:
    for p in [f["canonical_prompt"]] + f["same_template"] + f["related"]:
        assert not any(s in p for s in ("Atlantis", "France", "Germany")), p
assert len({t for _, _, t in FACTS}) == 12

bank = {
    "version": 1,
    "excluded_exploratory": sorted(SEEN),
    "facts": facts_out,
    "reserve": reserve_out,
    "unrelated": unrelated,
    "atomicity_probe": unrelated[::4][:50],
}
out = pathlib.Path(__file__).with_name("bank.json")
data = json.dumps(bank, indent=2, ensure_ascii=False) + "\n"
out.write_text(data)
print(hashlib.sha256(data.encode()).hexdigest(), len(unrelated), len(bank["atomicity_probe"]))
