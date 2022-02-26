# -*- coding: utf-8 -*-
"""
Created 2021-06-01

@author: Nathan Young

Adapted from PARARULE Plus Depth-5 generation code; Copyright (c) 2021, Qiming Bao. All rights reserved.
Thanks to Qiming Bao for development of the PARARULE Plus dataset and permission to freely utilise it.

Generates AbductionRules datasets and creates train/dev/test splits thereof.

"""

import itertools
import json
import random
import os

datasets = [
    "Abduction-Animal-Simple",
    "Abduction-Animal",
    "Abduction-Person-Simple",
    "Abduction-Person",
]


def random_bit():
    return random.randint(0, 1)


def sentencify(sentence):
    """Ensures a capital letter at the beginning and a full stop at the end of a given sentence."""
    sentence = list(sentence)
    sentence[0] = sentence[0].upper()
    if sentence[-1] != ".":
        sentence.append(".")
    sentence = "".join(sentence)
    return sentence


def build_context_fact(entity: str, attribute: str, value: bool = True):
    # Utimately an exercise in filler language
    colourful = random_bit()
    colourful2 = random_bit()
    positives = [  # TODO: Make these good instead of crap
        "is definitely",
        "definitely is",
        "is certainly",
        "certainly is",
        "is absolutely",
        "absolutely is",
        "can't not be",
        "is known to be",
        "is actually",
        "has always been",
        "is always",
    ]
    negatives = [
        "isn't",
        "can't be",
        "cannot be",
        "definitely isn't",
        "definitely is not",
        "is definitely not",
        "can't possibly be",
        "couldn't possibly be",
    ]
    if colourful:
        relation_set = [negatives, positives][value]
        relation = random.choice(relation_set)
    else:
        relation = ["is not", "is"][value]
    if colourful2:
        if value:
            modifier = random.choice(
                [
                    "extremely",
                    "very",
                    "quite",
                    "really",
                ]
            )
        else:
            modifier = random.choice(
                [
                    "at all",
                    "remotely",
                ]
            )
        line = f"{entity} {relation} {modifier} {attribute}"
    else:
        line = f"{entity} {relation} {attribute}"

    line = sentencify(line)
    return line
    # Use for answers???


def build_attributes(*attributes):
    # THis is where the "is" (or "are") should be taken out, not in the rule
    # Also, this is where the 'also' should be decided? Except it can be outside and it can't be both
    if len(attributes) == 1:
        return attributes[0]
    all_is = True
    for attribute in attributes:
        if attribute[:3] != "is " and attribute[:4] != "are ":
            all_is = False
            break
    if random_bit() and all_is:
        attributes = list(attributes)
        for i in range(1, len(attributes)):
            attributes[i] = attributes[i].removeprefix("is ")
            attributes[i] = attributes[i].removeprefix("are ")
    if len(attributes) == 2:
        return " and ".join(attributes)
    return ", ".join(attributes[:-1]) + ", and " + attributes[-1]


def build_rule(head_attributes, tail_attributes, domain):
    # TODO: Rename variables for deduction-inclusion
    # Add prefixes - "everyone knows that", "it's true that", "Surprisingly, " etc
    # Similar suffixes
    # Random modifiers - "is" -> "is always", "is certainly", etc
    # Remember to train a thing to do this as well
    # head_attributes = [attribute1, attribute2, attribute3]
    # all_attributes = [attribute1, attribute2, attribute3, to_abduce]
    if type(head_attributes) == str:
        head_attributes = [head_attributes]
    elif type(head_attributes) != list:
        head_attributes = list(head_attributes)
    if type(tail_attributes) == str:
        tail_attributes = [tail_attributes]
    elif type(tail_attributes) != list:
        tail_attributes = list(tail_attributes)
    random.shuffle(head_attributes)
    random.shuffle(tail_attributes)
    plural = random_bit()
    specific = random_bit()
    also = random_bit()
    multiplier = int(domain=="person") + 1
    inverted = random_bit()
    if also:
        if tail_attributes[0][0:2] == "is" and not inverted:
            as_list = tail_attributes[0].split()
            as_list.insert(1, "also")
            tail_attributes[0] = " ".join(as_list)
            also_text = ""
        elif inverted and head_attributes[0][0:2] == "is":
            as_list = head_attributes[0].split()
            as_list.insert(1, "also")
            head_attributes[0] = " ".join(as_list)
            also_text = ""
        else:
            also_text = "also "
    else:
        also_text = ""
    if plural:
        for attributes in [head_attributes, tail_attributes]:
            for index, attribute in enumerate(attributes):
                if attribute[0:2] == "is":
                    attribute = "are" + attribute[2:]
                else:
                    attribute = list(attribute)
                    space = attribute.index(" ")
                    attribute.pop(space - 1)
                    attribute = "".join(attribute)
                attributes[index] = attribute
        identifier = ["things", "animals", "people"][specific * multiplier]
        all = "all " * random_bit()  # TODO: every, everything, everyone
        if not inverted:
            line = f"{all}{identifier} that {build_attributes(*head_attributes)}, {also_text}{build_attributes(*tail_attributes)}."
        else:
            line = f"{identifier} {build_attributes(*tail_attributes)} if they {also_text}{build_attributes(*head_attributes)}."

    else:
        identifier = ["something", "an animal", "someone"][specific * multiplier]
        then = "then " * random_bit()
        identifier2 = ["it", "it", "that person"][
            specific * multiplier
        ]  # TODO: They, it's, they're - 'they' needs to be plural (stupid English pronouns)
        if not inverted:
            line = f"If {identifier} {build_attributes(*head_attributes)}, {then}{identifier2} {also_text}{build_attributes(*tail_attributes)}."
        else:
            line = f"{identifier} {build_attributes(*tail_attributes)} if {identifier2} {also_text}{build_attributes(*head_attributes)}."

    # TODO: Every possible contraction (it is) has a chance of getting contracted?

    line = sentencify(line)
    return line


animal_names = [
    # TODO: Remove "the"?
    "the bald eagle",
    "the tiger",
    "the bear",
    "the lion",
    "the wolf",
    "the crocodile",
    "the dinosaur",
    "the snake",
    "the leopard",
    "the cheetah",
    "the falcon",
    "the fox",
    "the panther",
    "the jaguar",
    # "the alligator"
    # the shark??
]
animal_names_1 = [
    "the cat",
    "the dog",
    "the mouse",
    "the rabbit",
    "the squirrel",
    "the hamster",
    # "the deer",
    # "the cow",
]
people_names = [
    "Anne",
    "Alan",
    "Bob",
    "Charlie",
    "Dave",
    "Erin",
    "Fiona",
    "Gary",
    "Harry",
]  # TODO: Expand?

relations = [
    "is",
    "is not",
]
animal_relations = ["likes", "chases", "needs", "visits", "attacks", "sees"]
animal_relations_1 = {
    "does not like",
    "does not chase",
    "does not need",
    "does not visit",
    "does not attack",
    "does not see",
}
animal_attributes_1 = ["kind", "quiet", "round", "nice", "smart"]
animal_attributes_2 = [
    "dull",
    "rough",
    "lazy",
    "slow",
    "sleepy",
    "tired",
    "reckless",
    "boring",
    "angry",
]
animal_attributes_3 = ["big", "strong", "awful", "fierce", "heavy", "obese"]
animal_attributes_4 = [
    "furry",
    "small",
    "cute",
    "lovely",
    "beautiful",
    "adorable",
    "funny",
]

people_attributes_1 = ["big", "strong", "high", "huge", "heavy"]
people_attributes_2 = ["short", "thin", "small", "little", "tiny"]
people_attributes_3 = ["wealthy", "smart", "nice", "quiet", "kind", "clever"]
people_attributes_4 = ["poor", "dull", "rough", "bad", "sad", "imperfect"]
people_attributes_5 = ["old"]
people_attributes_6 = ["young"]


def rules_name(logic, domain, extra_rules):
    return logic.capitalize() + "-" + domain.capitalize() + "-Simple" * (not extra_rules)


def generate_dataset(logic="abduction", domain="animal", extra_rules=False):
    name = rules_name(logic, domain, extra_rules)
    random.seed(name)
    # Proposal: just make categories of background facts, rules, and non-background facts, take one category out and make them into questions instead (only one abduction fact of course)
    # Take out random abduction facts!!!
    # Either have separate names or randomly assign different instances to de/in/abduction datasets (increase numbers of names to compensate)
    if domain == "animal":
        attributes = [
            animal_attributes_1.copy(),
            animal_attributes_2.copy(),
            animal_attributes_3.copy(),
            animal_attributes_4.copy(),
        ]
        items = [
            list(itertools.chain.from_iterable(item))
            for item in itertools.product(
                itertools.permutations(animal_names, 2),
                itertools.permutations(animal_names_1, 2),
            )
        ]
        relations_2 = animal_relations.copy()
        random.shuffle(relations_2)
    elif domain == "person":
        attributes = [
            people_attributes_1.copy(),
            people_attributes_2.copy(),
            people_attributes_3.copy(),
            people_attributes_4.copy(),
        ]
        items = itertools.permutations(people_names, 4)

    whole_dict = []
    for entry_id, names in enumerate(items):
        for attribute_list in attributes:
            random.shuffle(attribute_list)

        facts = [
            build_context_fact(names[n], attributes[[1, 2, 0, 3][n]][i])
            for n in range(4)
            for i in range(3)
        ]

        rules = [
            build_rule(
                [f"is {attributes[i][j]}" for j in [0, 1, 3]],
                f"is {attributes[i][4]}",
                domain,
            )
            for i in range(4)
        ]

        context = facts + rules

        questions = [
            sentencify(f"{names[n]} {relations[r]} {attributes[[1, 2, 0, 3][n]][4]}")
            for n in range(4)
            for r in range(2)
        ]

        labels = [
            sentencify(f"{names[n]} {relations[0]} {attributes[[1, 2, 0, 3][n]][3]}")
            for n in range(4)
            for _ in range(2)
        ]

        if domain == "animal":
            context += [
                f"{names[0]} {relations_2[0]} {names[2]}.",
                f"{names[1]} {relations_2[2]} {names[3]}.",
                build_rule(
                    [
                        f"is {attributes[1][2]}",
                        f"{relations_2[0]} {names[2]}",
                        f"{relations_2[1]} {names[3]}",
                    ],
                    f"is {attributes[2][5]}",
                    domain,
                ),
                # Extra - could be used to make a question
                build_rule(
                    [
                        f"is {attributes[2][2]}",
                        f"{relations_2[3]} {names[2]}",
                        f"{relations_2[2]} {names[3]}",
                    ],
                    f"is {attributes[1][5]}",
                    domain,
                ),
            ]

            questions += [
                sentencify(f"{names[0]} {relations[0]} {attributes[2][5]}"),
                sentencify(f"{names[0]} {relations[1]} {attributes[2][5]}"),
            ]

            labels += [sentencify(f"{names[0]} {relations_2[1]} {names[3]}")] * 2

        if extra_rules:
            # TODO: Make these only have a chance of showing up, so that they can't be relied on
            # TODO: Make these change to random attribute lists
            context += [
                build_rule(
                    [
                        f"is {attributes[i][0]}",
                        f"is {attributes[(i+1)%4][1]}",
                        f"is {attributes[i][3]}",
                    ],
                    f"is {attributes[i][4]}",
                    domain,
                )
                for i in range(4)
            ]

            context += [
                build_rule(
                    [
                        f"is {attributes[i][0]}",
                        f"is {attributes[(i+1)%4][1]}",
                        f"is {attributes[(i+2)%4][3]}",
                    ],
                    f"is {attributes[(i+3)%4][4]}",
                    domain,
                )
                for i in range(4)
            ]

        for index in range(len(context)):
            context[index] = sentencify(context[index])
        random.shuffle(context)

        test_dict = {
            "id": f"{name}-{entry_id}",
            "context": " ".join(context),
            "questions": [
                {
                    "id": f"{name}-{entry_id}-Q{index+1}",
                    "text": questions[index],
                    "label": labels[index],
                    "QCat": ["0", "0_0"][index % 2],
                }
                for index in range(len(questions))
            ],
        }

        whole_dict.append(test_dict)
    return whole_dict


def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError


def write_to_file(name, set, items):
    if not os.path.exists("datasets"):
        os.mkdir("datasets")
    if not os.path.exists(os.path.join("datasets", name)):
        os.mkdir(os.path.join("datasets", name))
    with open(os.path.join("datasets", name, f"{set}.jsonl"), "w") as file:
        for item in items:
            json.dump(item, file, default=set_default)
            file.write("\n")


def main():
    dataset_split = [7, 1, 2]  # train, dev, test

    for domain in ["animal", "person"]:
        for extra_rules in [False, True]:
            dataset = generate_dataset("abduction", domain, extra_rules)
            random.shuffle(dataset)

            per_unit = len(dataset) // sum(dataset_split)
            train = dataset_split[0] * per_unit
            train_dev = sum(dataset_split[:2]) * per_unit
            train_set = dataset[:train]
            dev_set = dataset[train:train_dev]
            test_set = dataset[train_dev:]

            name = rules_name("abduction", domain, extra_rules)
            write_to_file(name, "train", train_set)
            write_to_file(name, "dev", dev_set)
            write_to_file(name, "test", test_set)


if __name__ == "__main__":
    main()
