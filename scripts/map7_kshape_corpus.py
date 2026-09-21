#!/usr/bin/env python3
"""map7-kshape corpus generator (corpus_version map7-kshape-corpus-v1).

Deterministic generator for the shared MAP-7 / K-SHAPE corpus specified in
docs/preregistration/map7-kshape-capture-spec.md §4. Stdlib only. Running it
twice produces byte-identical output; the manifest records the SHA-256 of the
prompt file so every downstream consumer can refuse on drift.

Strata (5,000 prompts total):
  general      2,660  combinatorial open-ended continuations (incl. 160
                      backfill from the entity-swap holdout interaction)
  entity-swap    840  10 domains x 5 phrasings x 20 entities/domain, minus
                      the 160 cells dropped by the test-only-family /
                      entity-disjointness interaction (spec §4.1)
  templated      500  deliberately low-entropy continuations
  code           500  code continuations
  non-english    500  fr/de/es/it/pt prose continuations

Split: 60/20/20 at prompt level; entity-swap split follows the entity split
(12/4/4 per domain); one phrasing per domain is test-only and keeps only
test entities. Leakage rule: no prompt text may contain any entity-swap
canonical answer (case-insensitive, word/phrase boundary); candidates that
violate it are skipped deterministically.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

CORPUS_VERSION = "map7-kshape-corpus-v1"
SEED = 20260828
OUT_DIR = Path(__file__).resolve().parent.parent / "bench" / "map7-kshape" / "corpus-v1"

N_GENERAL = 2660
N_TEMPLATED = 500
N_CODE = 500
N_NONENGLISH = 500
MIN_WORDS = 24
ENTITIES_PER_DOMAIN = 20
ENTITY_SPLIT = (12, 4, 4)  # train/dev/test entities per domain
SPLIT_FRACTIONS = (0.6, 0.2, 0.2)
SPLIT_NAMES = ("train", "dev", "test")

# ---------------------------------------------------------------------------
# Entity-swap domains: (domain, carrier, [5 phrasings], [(entity, answer)])
# Answers must be distinctive strings; every answer joins the leakage ban set.
# ---------------------------------------------------------------------------

CARRIER_QUIZ = (
    "Here is an item taken from a general-knowledge quiz booklet. Read the "
    "item carefully, think about the fact it asks for, and complete the final "
    "sentence with the correct answer."
)

DOMAINS = [
    (
        "country-capital",
        CARRIER_QUIZ,
        [
            "The capital city of {e} is",
            "According to the atlas, the capital of {e} is",
            "Anyone studying geography learns that the capital of {e} is",
            "On the political map, {e} has its capital at",
            "The seat of government of {e} is the city of",
        ],
        [
            ("France", "Paris"), ("Japan", "Tokyo"), ("Egypt", "Cairo"),
            ("Canada", "Ottawa"), ("Australia", "Canberra"),
            ("Brazil", "Brasilia"), ("Kenya", "Nairobi"), ("Norway", "Oslo"),
            ("Thailand", "Bangkok"), ("Spain", "Madrid"),
            ("Poland", "Warsaw"), ("Turkey", "Ankara"),
            ("Argentina", "Buenos Aires"), ("Iran", "Tehran"),
            ("Vietnam", "Hanoi"), ("Portugal", "Lisbon"),
            ("Greece", "Athens"), ("Finland", "Helsinki"),
            ("Chile", "Santiago"), ("Ethiopia", "Addis Ababa"),
        ],
    ),
    (
        "country-continent",
        CARRIER_QUIZ,
        [
            "The country of {e} is located on the continent of",
            "Geographers place {e} on the continent called",
            "If you search for {e} on a globe, you will find it in",
            "The nation of {e} belongs to the continent of",
            "In every world atlas, {e} appears within the continent of",
        ],
        [
            ("Nigeria", "Africa"), ("Germany", "Europe"), ("India", "Asia"),
            ("Peru", "South America"), ("Mexico", "North America"),
            ("Sweden", "Europe"), ("Indonesia", "Asia"),
            ("Morocco", "Africa"), ("Ukraine", "Europe"),
            ("Colombia", "South America"), ("Mongolia", "Asia"),
            ("Algeria", "Africa"), ("Netherlands", "Europe"),
            ("Malaysia", "Asia"), ("Ghana", "Africa"),
            ("Bolivia", "South America"), ("Nepal", "Asia"),
            ("Tunisia", "Africa"), ("Austria", "Europe"),
            ("Cambodia", "Asia"),
        ],
    ),
    (
        "country-currency",
        CARRIER_QUIZ,
        [
            "The official currency of {e} is the",
            "Travellers exchanging money in {e} receive the local currency, the",
            "Prices in {e} are quoted in the national currency, the",
            "The unit of money used throughout {e} is called the",
            "Banks in {e} settle everyday accounts in the",
        ],
        [
            ("Japan", "yen"), ("India", "rupee"), ("Russia", "ruble"),
            ("Sweden", "krona"), ("Switzerland", "franc"),
            ("Israel", "shekel"), ("Thailand", "baht"), ("Vietnam", "dong"),
            ("Poland", "zloty"), ("Turkey", "lira"), ("Hungary", "forint"),
            ("Czech Republic", "koruna"), ("Bangladesh", "taka"),
            ("Saudi Arabia", "riyal"), ("Malaysia", "ringgit"),
            ("Indonesia", "rupiah"), ("Denmark", "krone"),
            ("Jordan", "dinar"), ("Mexico", "peso"), ("Morocco", "dirham"),
        ],
    ),
    (
        "book-author",
        (
            "Here is an item from a literature quiz for secondary-school "
            "students. Recall the classic work being described and complete "
            "the final sentence with the correct name."
        ),
        [
            'The novel "{e}" was written by',
            'The classic work "{e}" is the creation of the author',
            'Literature students know that "{e}" was authored by',
            'On the spine of any edition of "{e}" you will find the name',
            'The book "{e}" is universally attributed to',
        ],
        [
            ("Pride and Prejudice", "Austen"), ("Moby-Dick", "Melville"),
            ("War and Peace", "Tolstoy"),
            ("Crime and Punishment", "Dostoevsky"),
            ("Great Expectations", "Dickens"),
            ("The Great Gatsby", "Fitzgerald"),
            ("Don Quixote", "Cervantes"), ("Jane Eyre", "Bronte"),
            ("The Old Man and the Sea", "Hemingway"),
            ("Dracula", "Stoker"), ("Frankenstein", "Shelley"),
            ("The Trial", "Kafka"), ("Brave New World", "Huxley"),
            ("Lolita", "Nabokov"), ("Hamlet", "Shakespeare"),
            ("Paradise Lost", "Milton"), ("Robinson Crusoe", "Defoe"),
            ("Gulliver's Travels", "Swift"), ("The Odyssey", "Homer"),
            ("One Hundred Years of Solitude", "Marquez"),
        ],
    ),
    (
        "country-language",
        CARRIER_QUIZ,
        [
            "The primary language spoken in {e} is",
            "Most residents of {e} grow up speaking",
            "The national language of {e} is",
            "Street signs and newspapers in {e} are written in",
            "Visitors to {e} quickly notice that the local language is",
        ],
        [
            ("Japan", "Japanese"), ("Vietnam", "Vietnamese"),
            ("Greece", "Greek"), ("Thailand", "Thai"),
            ("Finland", "Finnish"), ("Hungary", "Hungarian"),
            ("Poland", "Polish"), ("Turkey", "Turkish"),
            ("Netherlands", "Dutch"), ("Sweden", "Swedish"),
            ("Israel", "Hebrew"), ("Iran", "Persian"),
            ("Ethiopia", "Amharic"), ("Cambodia", "Khmer"),
            ("Georgia", "Georgian"), ("Albania", "Albanian"),
            ("Denmark", "Danish"), ("Croatia", "Croatian"),
            ("Iceland", "Icelandic"), ("Mongolia", "Mongolian"),
        ],
    ),
    (
        "us-state-capital",
        (
            "Here is an item from an American civics quiz. Recall the state "
            "government facts you learned in school and complete the final "
            "sentence with the correct city."
        ),
        [
            "The capital of the state of {e} is",
            "The {e} state legislature meets in the capital city of",
            "Students of American geography know the capital of {e} is",
            "The governor of {e} works in the state capital,",
            "Official state business in {e} is conducted in",
        ],
        [
            ("California", "Sacramento"), ("Texas", "Austin"),
            ("Florida", "Tallahassee"), ("New York", "Albany"),
            ("Ohio", "Columbus"), ("Illinois", "Springfield"),
            ("Arizona", "Phoenix"), ("Colorado", "Denver"),
            ("Oregon", "Salem"), ("Washington", "Olympia"),
            ("Nevada", "Carson City"), ("Utah", "Salt Lake City"),
            ("Massachusetts", "Boston"), ("Michigan", "Lansing"),
            ("Minnesota", "Saint Paul"), ("Louisiana", "Baton Rouge"),
            ("Tennessee", "Nashville"), ("Kentucky", "Frankfort"),
            ("Alaska", "Juneau"), ("Georgia", "Atlanta"),
        ],
    ),
    (
        "atomic-number-element",
        (
            "Here is an item from a chemistry quiz. Recall the periodic "
            "table you studied and complete the final sentence with the "
            "correct element name."
        ),
        [
            "The chemical element with atomic number {e} is",
            "On the periodic table, atomic number {e} belongs to",
            "Chemistry students know element number {e} as",
            "The element occupying position {e} of the periodic table is",
            "Atomic number {e} identifies the element called",
        ],
        [
            ("1", "hydrogen"), ("2", "helium"), ("3", "lithium"),
            ("6", "carbon"), ("7", "nitrogen"), ("8", "oxygen"),
            ("9", "fluorine"), ("10", "neon"), ("11", "sodium"),
            ("12", "magnesium"), ("15", "phosphorus"), ("16", "sulfur"),
            ("17", "chlorine"), ("18", "argon"), ("19", "potassium"),
            ("20", "calcium"), ("22", "titanium"), ("24", "chromium"),
            ("25", "manganese"), ("28", "nickel"),
        ],
    ),
    (
        "animal-young",
        (
            "Here is an item from a nature quiz about animal families. "
            "Recall the standard English word for the baby animal and "
            "complete the final sentence with it."
        ),
        [
            "A baby {e} is called a",
            "Farmers and naturalists refer to a young {e} as a",
            "The standard English word for a newborn {e} is",
            "In wildlife documentaries, a juvenile {e} is called a",
            "The offspring of a {e} is known as a",
        ],
        [
            ("cow", "calf"), ("dog", "puppy"), ("cat", "kitten"),
            ("sheep", "lamb"), ("horse", "foal"), ("deer", "fawn"),
            ("bear", "cub"), ("kangaroo", "joey"), ("swan", "cygnet"),
            ("frog", "tadpole"), ("hen", "chick"), ("pig", "piglet"),
            ("duck", "duckling"), ("goose", "gosling"), ("owl", "owlet"),
            ("hare", "leveret"), ("eagle", "eaglet"), ("fox", "kit"),
            ("seal", "pup"), ("salmon", "smolt"),
        ],
    ),
    (
        "element-symbol-name",
        (
            "Here is an item from a chemistry quiz. Recall which element "
            "the chemical symbol stands for and complete the final sentence "
            "with the element's full name."
        ),
        [
            "The chemical symbol {e} stands for the element",
            "On the periodic table, the symbol {e} denotes",
            "Chemists reading the symbol {e} know it means",
            "In any chemical formula, {e} is the symbol for",
            "The two-letter symbol {e} identifies the element",
        ],
        [
            ("Fe", "iron"), ("Cu", "copper"), ("Zn", "zinc"),
            ("Ag", "silver"), ("Sn", "tin"), ("Au", "gold"),
            ("Hg", "mercury"), ("Pb", "lead"), ("Pt", "platinum"),
            ("Co", "cobalt"), ("Ra", "radium"), ("Rn", "radon"),
            ("Kr", "krypton"), ("Xe", "xenon"), ("Ba", "barium"),
            ("Sr", "strontium"), ("Br", "bromine"), ("Si", "silicon"),
            ("Bi", "bismuth"), ("Cs", "cesium"),
        ],
    ),
    (
        "river-continent",
        CARRIER_QUIZ,
        [
            "The river {e} flows across the continent of",
            "Geographers locate the {e} river on the continent of",
            "The great waterway known as the {e} runs through",
            "Every atlas shows the {e} river within the continent of",
            "The drainage basin of the {e} lies in",
        ],
        [
            ("Nile", "Africa"), ("Amazon", "South America"),
            ("Danube", "Europe"), ("Mekong", "Asia"),
            ("Mississippi", "North America"), ("Rhine", "Europe"),
            ("Ganges", "Asia"), ("Congo", "Africa"), ("Volga", "Europe"),
            ("Yangtze", "Asia"), ("Zambezi", "Africa"), ("Seine", "Europe"),
            ("Thames", "Europe"), ("Tigris", "Asia"),
            ("Euphrates", "Asia"), ("Loire", "Europe"),
            ("Niger", "Africa"), ("Orinoco", "South America"),
            ("Yukon", "North America"), ("Indus", "Asia"),
        ],
    ),
]

# ---------------------------------------------------------------------------
# General open-ended bank: openers x subjects x frames (combinatorial).
# ---------------------------------------------------------------------------

GEN_OPENERS = [
    "Thinking about this topic for a moment,",
    "After reading several long articles on the subject,",
    "During a conversation with an old friend last weekend,",
]

GEN_SUBJECTS = [
    "the history of tea cultivation", "urban beekeeping",
    "the design of medieval bridges", "long-distance sailing",
    "the economics of small bakeries", "restoring antique clocks",
    "the psychology of habit formation", "community theatre",
    "the future of public libraries", "mountain weather patterns",
    "the craft of letterpress printing", "night-sky photography",
    "the ecology of tidal pools", "learning a musical instrument as an adult",
    "the logistics of running a marathon", "traditional boat building",
    "the etiquette of formal dinners", "the revival of vinyl records",
    "growing vegetables on a balcony", "the maintenance of steam locomotives",
    "the sociology of open-plan offices", "amateur radio operation",
    "the preservation of old films", "the training of guide dogs",
    "the architecture of railway stations", "hand-made paper production",
    "the culture of early morning swimming", "repairing mechanical watches",
    "the planning of city parks", "the tradition of oral storytelling",
    "the chemistry of bread dough", "cross-country skiing technique",
    "the design of board games", "the daily routine of lighthouse keepers",
    "the market for second-hand books", "the acoustics of concert halls",
    "field sketching for naturalists", "the discipline of daily journaling",
    "the engineering of suspension footbridges", "cheese-making at home",
    "the migration of songbirds", "the restoration of stone walls",
    "the language of nautical flags", "the practice of bonsai",
    "the organisation of village festivals", "wilderness first aid",
    "the manufacture of pencils", "the layout of formal gardens",
    "the training of young referees", "the physics of kite flying",
    "the tradition of apprenticeship in carpentry", "map reading without GPS",
    "the running of a food cooperative", "the art of calligraphy",
    "the maintenance of canal locks", "backyard astronomy clubs",
    "the sourcing of natural dyes", "the routine of professional bakers",
    "the upkeep of hiking trails", "the etiquette of chess tournaments",
    "the design of school playgrounds", "fermenting vegetables safely",
    "the history of the bicycle", "the work of court stenographers",
    "the repair of woodwind instruments", "the planning of long rail journeys",
    "the culture of allotment gardening", "the craft of bookbinding",
    "the operation of small ferries", "the routines of orchestra rehearsal",
    "the stocking of village shops", "the art of topiary",
]

GEN_FRAMES = [
    "I have come to believe that the most overlooked aspect of {s} is",
    "what surprises newcomers to {s} more than anything else is",
    "the single biggest misconception people hold about {s} is",
    "if I had to explain {s} to a complete beginner, I would start with",
    "the part of {s} that rewards patience most generously is",
    "experienced practitioners of {s} tend to agree that",
    "the hardest lesson anyone learns about {s} is",
    "over the last decade, the practice of {s} has changed because",
    "the quiet satisfaction that comes from {s} arises mostly from",
    "a sensible first step for anyone curious about {s} is",
    "the reason {s} attracts such devoted followers is",
    "people who abandon {s} after a few weeks usually do so because",
    "the equipment needed for {s} matters far less than",
    "the seasonal rhythm of {s} means that",
    "written guides to {s} rarely mention that",
    "the community that has grown up around {s} values",
    "the economics of {s} only make sense once you realise",
    "teaching {s} to teenagers reveals that",
    "the future of {s} probably depends on",
    "my grandfather's approach to {s} always emphasised",
]

# ---------------------------------------------------------------------------
# Templated / low-entropy bank.
# ---------------------------------------------------------------------------

TEMPLATED_PREAMBLE = (
    "Please continue the following well-known sequence or passage exactly "
    "as it normally appears, without adding any commentary, explanation, or "
    "extra punctuation of your own."
)

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
            "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
DISHES = [
    ("a plain omelette", ["eggs", "butter", "salt"]),
    ("pancakes", ["flour", "milk", "eggs"]),
    ("a basic tomato sauce", ["tomatoes", "garlic", "olive oil"]),
    ("shortbread", ["butter", "sugar", "flour"]),
    ("a simple vinaigrette", ["oil", "vinegar", "mustard"]),
    ("porridge", ["oats", "water", "salt"]),
    ("garlic bread", ["bread", "butter", "garlic"]),
    ("a fruit smoothie", ["bananas", "berries", "yogurt"]),
    ("mashed potatoes", ["potatoes", "butter", "milk"]),
    ("rice pudding", ["rice", "milk", "sugar"]),
]
PROVERB_STEMS = [
    "A stitch in time saves",
    "The early bird catches the",
    "Actions speak louder than",
    "A picture is worth a thousand",
    "When in Rome, do as the",
    "Do not count your chickens before they",
    "Every cloud has a silver",
    "Practice makes",
    "Too many cooks spoil the",
    "The grass is always greener on the other",
]

# ---------------------------------------------------------------------------
# Code bank.
# ---------------------------------------------------------------------------

CODE_FUNCS = [
    ("sum_even_numbers", "values", "return the sum of the even entries"),
    ("count_vowels", "text", "count the vowels in the given text"),
    ("reverse_words", "sentence", "reverse the order of the words"),
    ("find_maximum", "numbers", "return the largest number in the list"),
    ("is_palindrome", "word", "check whether the word reads the same backwards"),
    ("merge_sorted", "left, right", "merge two sorted lists into one"),
    ("running_total", "amounts", "produce the cumulative totals"),
    ("strip_duplicates", "items", "remove duplicates while keeping order"),
    ("celsius_to_fahrenheit", "degrees", "convert a temperature reading"),
    ("word_frequencies", "document", "map each word to how often it occurs"),
    ("chunk_list", "items, size", "split the list into fixed-size chunks"),
    ("flatten_nested", "rows", "flatten one level of nesting"),
    ("binary_search", "sorted_items, target", "locate the target index"),
    ("moving_average", "series, window", "smooth the series with a window"),
    ("parse_key_values", "line", "parse a line of key=value pairs"),
    ("clamp_value", "value, low, high", "restrict the value to a range"),
    ("longest_common_prefix", "words", "find the shared starting characters"),
    ("rotate_left", "items, steps", "rotate the list left by n steps"),
    ("digits_only", "raw", "keep only the numeric characters"),
    ("pairwise_sums", "numbers", "sum each adjacent pair"),
    ("title_case", "phrase", "capitalise each word in the phrase"),
    ("median_value", "samples", "return the middle value of the samples"),
    ("trim_outliers", "readings", "drop readings far from the mean"),
    ("interleave", "first, second", "alternate elements from two lists"),
    ("histogram_bins", "values, bins", "count values falling in each bin"),
]

CODE_LANGS = [
    ("python", "# {doc}\ndef {name}({args}):\n    "),
    ("javascript", "// {doc}\nfunction {name}({args}) {{\n  "),
    ("rust", "/// {doc}\nfn {name}({args}) {{\n    "),
    ("go", "// {doc}\nfunc {name}({args}) {{\n\t"),
]

CODE_PREAMBLE = (
    "The following snippet comes from a small utility module in a larger "
    "project. Continue writing the body of the function in the same "
    "language and style, matching the naming conventions shown."
)

# ---------------------------------------------------------------------------
# Non-English bank: 5 languages x 10 seeds x 10 prefixes = 500 exactly.
# ---------------------------------------------------------------------------

NONENG = {
    "fr": {
        "prefixes": [
            "Comme je le disais hier a mes collegues,",
            "Selon un long article paru ce printemps,",
            "Apres plusieurs annees d'observation attentive,",
            "D'apres les habitants du quartier,",
            "Au fil de mes lectures recentes,",
            "Comme le rappelle souvent ma voisine,",
            "En repensant a ce voyage d'automne,",
            "Selon les notes de mon carnet,",
            "Comme l'expliquait notre professeur,",
            "Apres une longue discussion en famille,",
        ],
        "seeds": [
            "le petit marche du samedi matin attire toujours une foule fidele, car chacun sait que les producteurs de la region y apportent",
            "la vieille bibliotheque municipale a rouvert ses portes apres des travaux considerables, et les lecteurs decouvrent maintenant",
            "le sentier qui longe la riviere devient magnifique des les premiers jours du printemps, lorsque les arbres commencent a",
            "la boulangerie du coin prepare chaque matin une fournee speciale dont le parfum se repand dans toute la rue et donne envie de",
            "le train de nuit qui traverse la montagne offre aux voyageurs patients un spectacle rare, celui des vallees qui s'eclairent",
            "les jardins partages du quartier ont transforme la vie des habitants, qui se retrouvent chaque soir pour discuter et",
            "le vieux cinema de la place a decide de projeter de nouveau les grands classiques, et les jeunes spectateurs semblent",
            "la fete du village exige des mois de preparation, car chaque famille tient a presenter quelque chose qui rappelle",
            "l'atelier de poterie ouvert l'an dernier ne desemplit pas, et les apprentis repartent chaque semaine avec",
            "le marche aux livres anciens du dimanche reserve parfois de vraies surprises a ceux qui prennent le temps de",
        ],
    },
    "de": {
        "prefixes": [
            "Wie ich neulich in einer Zeitschrift las,",
            "Nach vielen Jahren im selben Viertel,",
            "Wie meine Grossmutter immer sagte,",
            "Laut den Berichten der Nachbarn,",
            "Nach einem langen Spaziergang am Fluss,",
            "Wie unser Lehrer damals erklaerte,",
            "Nach den Notizen in meinem Tagebuch,",
            "Wie die alten Leute im Dorf erzaehlen,",
            "Nach dem Gespraech von gestern Abend,",
            "Wie man auf dem Wochenmarkt hoert,",
        ],
        "seeds": [
            "der kleine Buchladen an der Ecke hat seine Auslage neu geordnet, und wer genau hinsieht, entdeckt zwischen den Regalen",
            "die Werkstatt des alten Uhrmachers riecht nach Holz und Metall, und auf seinem Tisch liegen Dutzende winziger Teile, die",
            "der Gemeinschaftsgarten hinter dem Bahnhof ist im Sommer der lebendigste Ort des Viertels, weil dort jeden Abend",
            "die Baeckerei oeffnet schon vor Sonnenaufgang, und der Duft des frischen Brotes zieht durch die stille Strasse bis",
            "der Wanderweg ueber die Huegel verlangt gutes Schuhwerk, doch die Aussicht am Gipfel belohnt jeden, der",
            "das kleine Museum am Marktplatz zeigt in diesem Jahr eine Sammlung alter Landkarten, die zeigen, wie",
            "die Musikschule im Hinterhof probt jeden Donnerstag, und die Nachbarn oeffnen ihre Fenster, um",
            "der Wochenmarkt beginnt puenktlich um sieben, und die Haendler stapeln ihre Kisten mit einer Sorgfalt, die",
            "die alte Muehle am Bach wurde liebevoll restauriert, und Besucher koennen nun beobachten, wie",
            "der Winter in dieser Gegend ist lang und still, doch die Bewohner haben gelernt, die dunklen Monate mit",
        ],
    },
    "es": {
        "prefixes": [
            "Como decia mi abuelo cada verano,",
            "Segun un reportaje que lei hace poco,",
            "Despues de muchos anos en el barrio,",
            "Como cuentan los vecinos mas antiguos,",
            "Tras una larga caminata por la costa,",
            "Segun las notas de mi cuaderno,",
            "Como explicaba nuestra maestra,",
            "Despues de la charla de anoche,",
            "Como se comenta en la plaza,",
            "Segun la costumbre de mi familia,",
        ],
        "seeds": [
            "el mercado de los sabados llena la plaza de colores y voces, y los puestos de fruta compiten por ofrecer",
            "la biblioteca del pueblo guarda una coleccion de mapas antiguos que muestran como era la comarca cuando",
            "el sendero que sube al faro se vuelve precioso en primavera, cuando las flores silvestres cubren",
            "la panaderia de la esquina hornea desde el amanecer, y el aroma del pan recien hecho invita a",
            "el tren de la tarde cruza los campos de olivos con una calma que permite a los viajeros contemplar",
            "el taller de ceramica abrio hace un ano y ya no queda sitio libre, porque cada semana los aprendices",
            "la fiesta del barrio requiere meses de preparativos, y cada familia insiste en aportar algo que recuerde",
            "el viejo cine de la avenida volvio a proyectar peliculas clasicas, y los jovenes descubren ahora",
            "el huerto comunitario detras de la escuela ha cambiado la vida de la calle, porque cada tarde",
            "la libreria de segunda mano esconde tesoros para quien tenga paciencia de revisar",
        ],
    },
    "it": {
        "prefixes": [
            "Come raccontava sempre mio nonno,",
            "Secondo un articolo letto di recente,",
            "Dopo tanti anni nello stesso quartiere,",
            "Come dicono i vicini piu anziani,",
            "Dopo una lunga passeggiata al porto,",
            "Secondo gli appunti del mio taccuino,",
            "Come spiegava il nostro maestro,",
            "Dopo la discussione di ieri sera,",
            "Come si sente dire al mercato,",
            "Secondo la tradizione di famiglia,",
        ],
        "seeds": [
            "il mercato del sabato riempie la piazza di profumi e di voci, e i banchi della frutta fanno a gara per esporre",
            "la biblioteca comunale conserva una raccolta di carte antiche che mostrano come era la valle quando",
            "il sentiero che sale verso l'eremo diventa splendido in primavera, quando i ciliegi cominciano a",
            "il forno all'angolo sforna il pane prima dell'alba, e il profumo attraversa la strada fino a",
            "il treno locale attraversa le colline con una lentezza che permette ai viaggiatori di osservare",
            "il laboratorio di ceramica aperto l'anno scorso e sempre pieno, perche ogni settimana gli allievi",
            "la festa del rione richiede mesi di preparativi, e ogni famiglia vuole portare qualcosa che ricordi",
            "il vecchio cinema della piazza ha ripreso a proiettare i classici, e i ragazzi scoprono adesso",
            "l'orto condiviso dietro la scuola ha cambiato la vita della via, perche ogni sera i vicini",
            "la libreria dell'usato nasconde meraviglie per chi ha la pazienza di sfogliare",
        ],
    },
    "pt": {
        "prefixes": [
            "Como dizia o meu avo todos os veroes,",
            "Segundo uma reportagem que li ha pouco,",
            "Depois de muitos anos no mesmo bairro,",
            "Como contam os vizinhos mais antigos,",
            "Depois de uma longa caminhada na praia,",
            "Segundo as notas do meu caderno,",
            "Como explicava a nossa professora,",
            "Depois da conversa de ontem a noite,",
            "Como se ouve dizer na praca,",
            "Segundo o costume da minha familia,",
        ],
        "seeds": [
            "o mercado de sabado enche a praca de cores e vozes, e as bancas de fruta competem para mostrar",
            "a biblioteca municipal guarda uma colecao de mapas antigos que revelam como era a regiao quando",
            "o trilho que sobe ate ao farol fica lindo na primavera, quando as flores silvestres cobrem",
            "a padaria da esquina trabalha desde a madrugada, e o cheiro do pao quente convida a",
            "o comboio da tarde atravessa os campos com uma calma que permite aos viajantes observar",
            "a oficina de ceramica abriu ha um ano e esta sempre cheia, porque em cada semana os aprendizes",
            "a festa do bairro exige meses de preparativos, e cada familia faz questao de trazer algo que lembre",
            "o velho cinema da avenida voltou a exibir os classicos, e os mais novos descobrem agora",
            "a horta comunitaria atras da escola mudou a vida da rua, porque todas as tardes",
            "o alfarrabista da baixa esconde tesouros para quem tiver paciencia de folhear",
        ],
    },
}

NONENG_LEADIN = {
    "fr": "C'est un sujet qui m'accompagne depuis des annees et que je ne me lasse pas d'observer.",
    "de": "Es ist ein Thema, das mich seit vielen Jahren begleitet und immer wieder beschaeftigt.",
    "es": "Es un tema que me acompana desde hace muchos anos y que nunca deja de interesarme.",
    "it": "E un argomento che mi accompagna da molti anni e che non smette mai di interessarmi.",
    "pt": "E um tema que me acompanha ha muitos anos e que nunca deixa de me interessar.",
}

# ---------------------------------------------------------------------------
# Deterministic RNG (explicit LCG so the output never depends on Python's
# random module implementation details across versions).
# ---------------------------------------------------------------------------


class Lcg:
    """Numerical Recipes LCG; deterministic across platforms/versions."""

    def __init__(self, seed: int) -> None:
        self.state = seed & 0xFFFFFFFF

    def next(self) -> int:
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return self.state

    def below(self, n: int) -> int:
        return self.next() % n

    def shuffle(self, items: list) -> None:
        for i in range(len(items) - 1, 0, -1):
            j = self.below(i + 1)
            items[i], items[j] = items[j], items[i]


def word_count(text: str) -> int:
    return len(text.split())


def build_ban_regex() -> re.Pattern:
    answers = sorted({a for _, _, _, pairs in DOMAINS for _, a in pairs})
    parts = [r"\b" + re.escape(a).replace(r"\ ", r"\s+") + r"\b" for a in answers]
    return re.compile("|".join(parts), re.IGNORECASE)


def main() -> int:
    rng = Lcg(SEED)
    ban = build_ban_regex()
    records: list[dict] = []
    dropped_cells: list[str] = []

    # --- entity-swap stratum -------------------------------------------------
    for domain, carrier, phrasings, pairs in DOMAINS:
        entity_order = list(range(ENTITIES_PER_DOMAIN))
        rng.shuffle(entity_order)
        n_tr, n_dv, _ = ENTITY_SPLIT
        entity_split = {}
        for rank, idx in enumerate(entity_order):
            if rank < n_tr:
                entity_split[idx] = "train"
            elif rank < n_tr + n_dv:
                entity_split[idx] = "dev"
            else:
                entity_split[idx] = "test"
        test_only_phrasing = rng.below(len(phrasings))
        for p_idx, phrasing in enumerate(phrasings):
            family = f"{domain}-p{p_idx}"
            for e_idx, (entity, answer) in enumerate(pairs):
                split = entity_split[e_idx]
                if p_idx == test_only_phrasing and split != "test":
                    dropped_cells.append(f"{family}/{entity}")
                    continue
                text = f"{carrier} {phrasing.format(e=entity)}"
                records.append({
                    "prompt_id": f"es-{domain}-p{p_idx}-e{e_idx:02d}",
                    "stratum": "entity-swap",
                    "domain": domain,
                    "family": family,
                    "family_heldout": p_idx == test_only_phrasing,
                    "entity": entity,
                    "answer": answer,
                    "split": split,
                    "text": text,
                })

    # --- general stratum -----------------------------------------------------
    candidates = []
    for o_idx, opener in enumerate(GEN_OPENERS):
        for s_idx, subject in enumerate(GEN_SUBJECTS):
            for f_idx, frame in enumerate(GEN_FRAMES):
                candidates.append((o_idx, s_idx, f_idx,
                                   f"{opener} {frame.format(s=subject)}"))
    rng.shuffle(candidates)
    taken = 0
    for o_idx, s_idx, f_idx, text in candidates:
        if taken >= N_GENERAL:
            break
        if ban.search(text) or word_count(text) < MIN_WORDS - 12:
            continue
        pad = ("I keep returning to this because it connects several things "
               "I care about, and ")
        full = (f"{pad}{text[0].lower()}{text[1:]}"
                if word_count(text) < MIN_WORDS else text)
        if ban.search(full) or word_count(full) < MIN_WORDS:
            continue
        records.append({
            "prompt_id": f"gen-{o_idx}-{s_idx:02d}-{f_idx:02d}",
            "stratum": "general", "domain": None, "family": None,
            "family_heldout": None, "entity": None, "answer": None,
            "split": None, "text": full,
        })
        taken += 1
    if taken < N_GENERAL:
        print(f"REFUSE: general bank exhausted at {taken}/{N_GENERAL}")
        return 1

    # --- templated stratum ---------------------------------------------------
    templated: list[tuple[str, str]] = []
    for start in range(7):
        days = ", ".join(WEEKDAYS[start:start + 3])
        templated.append((f"tmpl-week-{start}",
                          f"The days of the week continue in order: {days},"))
    for start in range(12):
        ms = ", ".join(MONTHS[start:start + 3])
        templated.append((f"tmpl-month-{start}",
                          f"The months of the year continue in order: {ms},"))
    for n in range(2, 200):
        templated.append((f"tmpl-count-{n}",
                          f"Counting upwards from {n}, the sequence runs {n}, "
                          f"{n + 1}, {n + 2}, {n + 3},"))
    for n in range(30, 120):
        templated.append((f"tmpl-countdown-{n}",
                          f"Counting downwards from {n}, the sequence runs "
                          f"{n}, {n - 1}, {n - 2}, {n - 3},"))
    for n in range(2, 21):
        for length in range(3, 13):
            row = ", ".join(str(n * k) for k in range(1, length))
            templated.append((f"tmpl-times-{n}-{length}",
                              f"The {n} times table begins {row},"))
    for d_idx, (dish, ingredients) in enumerate(DISHES):
        for cut in (1, 2):
            listed = ", ".join(ingredients[:cut])
            templated.append((f"tmpl-dish-{d_idx}-{cut}",
                              f"The basic ingredients for {dish} are {listed},"))
    for p_idx, stem in enumerate(PROVERB_STEMS):
        templated.append((f"tmpl-proverb-{p_idx}",
                          f"As the old saying goes: {stem}"))
    taken = 0
    for tid, body in templated:
        if taken >= N_TEMPLATED:
            break
        text = f"{TEMPLATED_PREAMBLE} {body}"
        if ban.search(text) or word_count(text) < MIN_WORDS:
            continue
        records.append({
            "prompt_id": tid, "stratum": "templated", "domain": None,
            "family": None, "family_heldout": None, "entity": None,
            "answer": None, "split": None, "text": text,
        })
        taken += 1
    if taken < N_TEMPLATED:
        print(f"REFUSE: templated bank exhausted at {taken}/{N_TEMPLATED}")
        return 1

    # --- code stratum --------------------------------------------------------
    code_candidates = []
    for f_idx, (name, args, doc) in enumerate(CODE_FUNCS):
        for l_idx, (lang, skeleton) in enumerate(CODE_LANGS):
            for v in range(6):
                vname = name if v == 0 else f"{name}_v{v}"
                body = skeleton.format(doc=doc, name=vname, args=args)
                code_candidates.append(
                    (f"code-{lang}-{f_idx:02d}-{v}",
                     f"{CODE_PREAMBLE}\n\n{body}"))
    rng.shuffle(code_candidates)
    taken = 0
    for cid, text in code_candidates:
        if taken >= N_CODE:
            break
        if ban.search(text) or word_count(text) < MIN_WORDS:
            continue
        records.append({
            "prompt_id": cid, "stratum": "code", "domain": None,
            "family": None, "family_heldout": None, "entity": None,
            "answer": None, "split": None, "text": text,
        })
        taken += 1
    if taken < N_CODE:
        print(f"REFUSE: code bank exhausted at {taken}/{N_CODE}")
        return 1

    # --- non-english stratum -------------------------------------------------
    taken = 0
    for lang in sorted(NONENG):
        bank = NONENG[lang]
        for p_idx, prefix in enumerate(bank["prefixes"]):
            for s_idx, seed_text in enumerate(bank["seeds"]):
                text = f"{NONENG_LEADIN[lang]} {prefix} {seed_text}"
                if ban.search(text) or word_count(text) < MIN_WORDS:
                    print(f"REFUSE: non-english cell {lang}-{p_idx}-{s_idx} "
                          f"failed leakage/length check")
                    return 1
                records.append({
                    "prompt_id": f"ne-{lang}-{p_idx}-{s_idx}",
                    "stratum": "non-english", "domain": None, "family": None,
                    "family_heldout": None, "entity": None, "answer": None,
                    "split": None, "text": text,
                })
                taken += 1
    if taken != N_NONENGLISH:
        print(f"REFUSE: non-english stratum produced {taken}, "
              f"expected {N_NONENGLISH}")
        return 1

    # --- prompt-level splits for non-entity strata ---------------------------
    unsplit = [r for r in records if r["split"] is None]
    order = list(range(len(unsplit)))
    rng.shuffle(order)
    n = len(order)
    n_train = int(n * SPLIT_FRACTIONS[0])
    n_dev = int(n * SPLIT_FRACTIONS[1])
    for rank, idx in enumerate(order):
        if rank < n_train:
            unsplit[idx]["split"] = "train"
        elif rank < n_train + n_dev:
            unsplit[idx]["split"] = "dev"
        else:
            unsplit[idx]["split"] = "test"

    # --- self-checks (refuse-style: any failure aborts with no output) -------
    texts = [r["text"] for r in records]
    if len(set(texts)) != len(texts):
        print("REFUSE: duplicate prompt texts")
        return 1
    if len(set(r["prompt_id"] for r in records)) != len(records):
        print("REFUSE: duplicate prompt ids")
        return 1
    for r in records:
        if ban.search(r["text"]):
            print(f"REFUSE: leakage in {r['prompt_id']}")
            return 1
        if word_count(r["text"]) < MIN_WORDS:
            print(f"REFUSE: short prompt {r['prompt_id']}")
            return 1
        if r["split"] not in SPLIT_NAMES:
            print(f"REFUSE: unsplit prompt {r['prompt_id']}")
            return 1
    counts = {}
    for r in records:
        counts[r["stratum"]] = counts.get(r["stratum"], 0) + 1
    expected = {"general": N_GENERAL, "entity-swap": 840,
                "templated": N_TEMPLATED, "code": N_CODE,
                "non-english": N_NONENGLISH}
    if counts != expected:
        print(f"REFUSE: stratum counts {counts} != {expected}")
        return 1
    if len(records) != 5000:
        print(f"REFUSE: total {len(records)} != 5000")
        return 1
    if len(dropped_cells) != 160:
        print(f"REFUSE: dropped cells {len(dropped_cells)} != 160")
        return 1

    # --- output --------------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prompts_path = OUT_DIR / "prompts.jsonl"
    lines = [json.dumps(r, ensure_ascii=False, sort_keys=True) for r in records]
    prompts_bytes = ("\n".join(lines) + "\n").encode("utf-8")
    prompts_path.write_bytes(prompts_bytes)

    split_counts = {}
    for r in records:
        key = (r["stratum"], r["split"])
        split_counts[key] = split_counts.get(key, 0) + 1
    manifest = {
        "corpus_version": CORPUS_VERSION,
        "seed": SEED,
        "generator": "scripts/map7_kshape_corpus.py",
        "prompts_sha256": hashlib.sha256(prompts_bytes).hexdigest(),
        "total_prompts": len(records),
        "stratum_counts": expected,
        "split_counts": {f"{s}/{sp}": c
                         for (s, sp), c in sorted(split_counts.items())},
        "entity_split_per_domain": {"train": 12, "dev": 4, "test": 4},
        "test_only_families": "one phrasing per domain, LCG-chosen",
        "dropped_cells": sorted(dropped_cells),
        "min_words": MIN_WORDS,
        "position_rule": "final min(32, len-8) token positions; index<8 never sampled (capture spec 4.2)",
        "leakage_rule": "no prompt contains any entity-swap canonical answer (word/phrase boundary, case-insensitive)",
        "pending_at_bringup": ["model_digest", "vindex_digest",
                              "capture_harness_git_sha", "probe_layers"],
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"OK: {len(records)} prompts -> {prompts_path}")
    print(f"OK: sha256 {manifest['prompts_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
