# -*- coding: utf-8 -*-
"""
Created 2021-06-01
​
@author: Nathan Young

Adapted from PARARULE Plus Depth-5 generation code; Copyright (c) 2021, Qiming Bao. All rights reserved.
Thanks to Qiming Bao for development of the PARARULE Plus dataset and permission to freely utilise it.
​
Generates AbductionRules datasets and creates train/dev/test splits thereof.
​
"""

import itertools
import json
import random
import os

datasets = [
    "Abduction-Animal-0.1",
    "Abduction-Animal-0.2",
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


def build_context(attribute1, attribute2, attribute3, to_abduce: str, dataset_index):
    attributes = [attribute1, attribute2, attribute3]
    random.shuffle(attributes)
    plural = random_bit()
    specific = random_bit()
    also = random_bit()
    multiplier = int(dataset_index > 4) + 1
    if also:
        if to_abduce[0:2] == "is":
            as_list = to_abduce.split()
            as_list.insert(1, "also")
            to_abduce = " ".join(as_list)
        else:
            to_abduce = "also " + to_abduce
    attributes.append(to_abduce)
    if plural:
        fixed = []
        for attribute in attributes:
            if attribute[0:2] == "is":
                attribute = "are" + attribute[2:]
            else:
                attribute = list(attribute)
                space = attribute.index(" ")
                attribute.pop(space - 1)
                attribute = "".join(attribute)
            fixed.append(attribute)
        identifier = ["things", "animals", "people"][specific * multiplier]
        all = "all " * random_bit()
        line = f"{all}{identifier} that {fixed[0]}, {fixed[1]}, and {fixed[2]}, {fixed[3]}."
        line = sentencify(line)
    else:
        identifier = ["something", "an animal", "a person"][specific * multiplier]
        then = "then " * random_bit()
        identifier2 = ["it", "it", "that person"][specific * multiplier]
        line = f"If {identifier} {attributes[0]}, {attributes[1]}, and {attributes[2]}, {then}{identifier2} {attributes[3]}."

    return line


animal_names = [
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
    # 'the fox',
    # 'the panther',
]
animal_names_1 = [
    "the cat",
    "the dog",
    "the mouse",
    "the rabbit",
    "the squirrel",
    "the hamster",
    # 'the deer',
    # 'the cow',
]
people_names = [
    "Anne",
    "Alan",
    "Bob",
    "Charlie",
    "Dave",
    "Erin",
    "Harry",
    "Gary",
    "Fiona",
]

relations = ["is", "is not"]
animal_relations = ["likes", "chases", "needs", "visits", "attacks", "sees"]
# animal_relations_1 = {
#     "does not like",
#     "does not chase",
#     "does not need",
#     "does not visit",
#     "does not attack",
#     "does not see",
# }
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


def generate_dataset(dataset_index):

    if dataset_index <= 4:
        attributes = [
            [],
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
    else:
        attributes = [
            [],
            people_attributes_1.copy(),
            people_attributes_2.copy(),
            people_attributes_3.copy(),
            people_attributes_4.copy(),
        ]
        items = itertools.permutations(people_names, 4)
    relations_2 = animal_relations.copy()

    whole_dict = []
    for entry_id, names in enumerate(items):
        entry_id += 1

        random.shuffle(attributes[1])
        random.shuffle(attributes[2])
        random.shuffle(attributes[3])
        random.shuffle(attributes[4])
        random.shuffle(relations_2)

        main_relation = relations[0]

        context = [
            f"{names[0]} {relations[0]} {attributes[2][0]}.",
            f"{names[0]} {relations[0]} {attributes[2][1]}.",
            f"{names[0]} {relations[0]} {attributes[2][2]}.",
            f"{names[1]} {relations[0]} {attributes[3][0]}.",
            f"{names[1]} {relations[0]} {attributes[3][1]}.",
            f"{names[2]} {relations[0]} {attributes[1][0]}.",
            f"{names[2]} {relations[0]} {attributes[1][1]}.",
            f"{names[2]} {relations[0]} {attributes[1][2]}.",
            f"{names[3]} {relations[0]} {attributes[4][0]}.",
            f"{names[3]} {relations[0]} {attributes[4][1]}.",
            f"{names[3]} {relations[0]} {attributes[4][2]}.",
        ]

        if dataset_index <= 4:
            context.extend(
                [
                    f"{names[0]} {relations_2[0]} {names[2]}.",
                    f"{names[1]} {relations_2[2]} {names[3]}.",
                ]
            )

        if dataset_index <= 2:
            context.extend(
                [
                    f"Things that are {attributes[1][0]}, {attributes[1][1]}, and {attributes[1][3]} are also {attributes[1][4]}.",
                    f"If something is {attributes[2][0]}, {attributes[2][3]}, and {attributes[2][1]} then it is {attributes[2][4]}.",
                    f"If an animal is {attributes[3][3]}, {attributes[3][1]}, and {attributes[3][0]} then it is also {attributes[3][4]}.",
                    f"All animals that are {attributes[4][0]}, {attributes[4][1]}, and {attributes[4][3]} are also {attributes[4][4]}.",
                    f"If something is {attributes[2][2]}, {relations_2[0]} {names[2]}, and {relations_2[1]} {names[3]}, then it is {attributes[3][5]}.",
                    f"If something {relations_2[3]} {names[2]}, {relations_2[2]} {names[3]}, and is {attributes[3][2]}, then it is {attributes[2][5]}.",
                ]
            )

        if dataset_index >= 3:
            context.extend(
                [
                    build_context(
                        f"is {attributes[1][0]}",
                        f"is {attributes[1][1]}",
                        f"is {attributes[1][3]}",
                        f"is {attributes[1][4]}",
                        dataset_index,
                    ),
                    build_context(
                        f"is {attributes[2][0]}",
                        f"is {attributes[2][1]}",
                        f"is {attributes[2][3]}",
                        f"is {attributes[2][4]}",
                        dataset_index,
                    ),
                    build_context(
                        f"is {attributes[3][0]}",
                        f"is {attributes[3][1]}",
                        f"is {attributes[3][3]}",
                        f"is {attributes[3][4]}",
                        dataset_index,
                    ),
                    build_context(
                        f"is {attributes[4][0]}",
                        f"is {attributes[4][1]}",
                        f"is {attributes[4][3]}",
                        f"is {attributes[4][4]}",
                        dataset_index,
                    ),
                ]
            )

        if 3 <= dataset_index <= 4:
            context.extend(
                [
                    build_context(
                        f"is {attributes[2][2]}",
                        f"{relations_2[0]} {names[2]}",
                        f"{relations_2[1]} {names[3]}",
                        f"is {attributes[3][5]}",
                        dataset_index,
                    ),
                    # Extra but can be turned into a question
                    build_context(
                        f"is {attributes[3][2]}",
                        f"{relations_2[3]} {names[2]}",
                        f"{relations_2[2]} {names[3]}",
                        f"is {attributes[2][5]}",
                        dataset_index,
                    ),
                ]
            )

        if dataset_index in [4, 6]:
            context.extend(
                [
                    build_context(
                        f"is {attributes[1][0]}",
                        f"is {attributes[2][1]}",
                        f"is {attributes[1][3]}",
                        f"is {attributes[1][4]}",
                        dataset_index,
                    ),
                    build_context(
                        f"is {attributes[2][0]}",
                        f"is {attributes[3][1]}",
                        f"is {attributes[2][3]}",
                        f"is {attributes[2][4]}",
                        dataset_index,
                    ),
                    build_context(
                        f"is {attributes[3][0]}",
                        f"is {attributes[4][1]}",
                        f"is {attributes[3][3]}",
                        f"is {attributes[3][4]}",
                        dataset_index,
                    ),
                    build_context(
                        f"is {attributes[4][0]}",
                        f"is {attributes[1][1]}",
                        f"is {attributes[4][3]}",
                        f"is {attributes[4][4]}",
                        dataset_index,
                    ),
                    build_context(
                        f"is {attributes[1][0]}",
                        f"is {attributes[2][1]}",
                        f"is {attributes[3][3]}",
                        f"is {attributes[4][4]}",
                        dataset_index,
                    ),
                    build_context(
                        f"is {attributes[4][0]}",
                        f"is {attributes[1][1]}",
                        f"is {attributes[2][3]}",
                        f"is {attributes[3][4]}",
                        dataset_index,
                    ),
                    build_context(
                        f"is {attributes[3][0]}",
                        f"is {attributes[4][1]}",
                        f"is {attributes[1][3]}",
                        f"is {attributes[2][4]}",
                        dataset_index,
                    ),
                    build_context(
                        f"is {attributes[2][0]}",
                        f"is {attributes[3][1]}",
                        f"is {attributes[4][3]}",
                        f"is {attributes[1][4]}",
                        dataset_index,
                    ),
                ]
            )

        for index in range(len(context)):
            context[index] = sentencify(context[index])
        if dataset_index > 1:
            random.shuffle(context)

        questions = [
            sentencify(f"{names[2]} {relations[0]} {attributes[1][4]}"),
            sentencify(f"{names[2]} {relations[1]} {attributes[1][4]}"),
            sentencify(f"{names[0]} {relations[0]} {attributes[2][4]}"),
            sentencify(f"{names[0]} {relations[1]} {attributes[2][4]}"),
            sentencify(f"{names[1]} {relations[0]} {attributes[3][4]}"),
            sentencify(f"{names[1]} {relations[1]} {attributes[3][4]}"),
            sentencify(f"{names[3]} {relations[0]} {attributes[4][4]}"),
            sentencify(f"{names[3]} {relations[1]} {attributes[4][4]}"),
        ]
        if dataset_index <= 4:
            questions.extend(
                [
                    sentencify(f"{names[0]} {relations[0]} {attributes[3][5]}"),
                    sentencify(f"{names[0]} {relations[1]} {attributes[3][5]}"),
                ]
            )

        labels = [
            sentencify(f"{names[2]} {relations[0]} {attributes[1][3]}"),
            sentencify(f"{names[2]} {relations[0]} {attributes[1][3]}"),
            sentencify(f"{names[0]} {relations[0]} {attributes[2][3]}"),
            sentencify(f"{names[0]} {relations[0]} {attributes[2][3]}"),
            sentencify(f"{names[1]} {relations[0]} {attributes[3][3]}"),
            sentencify(f"{names[1]} {relations[0]} {attributes[3][3]}"),
            sentencify(f"{names[3]} {relations[0]} {attributes[4][3]}"),
            sentencify(f"{names[3]} {relations[0]} {attributes[4][3]}"),
        ]
        if dataset_index <= 4:
            labels.extend(
                [
                    sentencify(f"{names[0]} {relations_2[1]} {names[3]}"),
                    sentencify(f"{names[0]} {relations_2[1]} {names[3]}"),
                ]
            )

        name = datasets[dataset_index - 1]
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


def write_to_file(index, set, items):
    name = datasets[index - 1]
    if not os.path.exists("datasets"):
        os.mkdir("datasets")
    if not os.path.exists(os.path.join("datasets", name)):
        os.mkdir(os.path.join("datasets", name))
    with open(os.path.join("datasets", name, f"{set}.jsonl"), "w") as file:
        for item in items:
            json.dump(item, file, default=set_default)
            file.write("\n")


def main(include_early=True):
    dataset_split = [7, 1, 2]  # train, dev, test

    if include_early:
        set_range = range(1, 7)
    else:
        set_range = range(3, 7)

    for i in set_range:
        random.seed(datasets[i - 1])
        dataset = generate_dataset(i)
        random.shuffle(dataset)

        per_unit = len(dataset) // sum(dataset_split)
        train = dataset_split[0] * per_unit
        train_dev = sum(dataset_split[:2]) * per_unit
        train_set = dataset[:train]
        dev_set = dataset[train:train_dev]
        test_set = dataset[train_dev:]

        write_to_file(i, "train", train_set)
        write_to_file(i, "dev", dev_set)
        write_to_file(i, "test", test_set)


if __name__ == "__main__":
    main()
