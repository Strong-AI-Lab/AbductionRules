import ast
import os
from collections import Counter

import generate

entities = generate.animal_names + generate.animal_names_1 + generate.people_names
attributes = (
    generate.people_attributes_1
    + generate.people_attributes_2
    + generate.people_attributes_3
    + generate.people_attributes_4
    + generate.people_attributes_5
    + generate.people_attributes_6
    + generate.animal_attributes_1
    + generate.animal_attributes_2
    + generate.animal_attributes_3
    + generate.animal_attributes_4
)
relations = generate.animal_relations


def as_percent(numerator, denominator, ndigits=1):
    return str(round(numerator / denominator * 100, ndigits)) + "%"


datasets = generate.datasets
models = datasets + [ "Person+Animal-0.1", "Animal+Person-Simple",]


def decompose(line: str):
    words = line.replace(".", "").replace("The", "the").split()
    subject = relation = attribute = obj = ""
    if "is" in words:
        subject = " ".join(words[0:-2])
        attribute = words[-1]
        relation = words[-2]
    else:
        for relation in relations:
            if relation in words:
                location = words.index(relation)
                subject = " ".join(words[0:location])
                obj = " ".join(words[location + 1 :])
                attribute = " ".join(words[location:])
                break

    return subject, relation, attribute, obj


def well_formed(line: str):
    try:
        subject, relation, attribute, obj = decompose(line)
    except IndexError:
        return False

    if relation == "is":
        return subject in entities and attribute in attributes
    else:
        return subject in entities and obj in entities and relation in relations


def attempt_to_fix(line0: str, line1: str):
    subject, relation, attribute, object = decompose(line0)

    # Extraneous words - remove
    newline = line1.replace(" and", "")
    newline = newline.replace(" are", " is")
    newline = newline.replace(" a ", " ")
    if not " is" in line0 and " is" in newline:
        newline = newline.replace(" is", "")
    line1 = newline

    # Looping
    words = line1.removesuffix(".").split()
    unique_words = set(words)
    for word in unique_words:
        index = words.index(word)
        while word in words:
            words.remove(word)
        words.insert(index, word)
    line1 = generate.sentencify(" ".join(words))

    # Second kind of looping; sometimes sentences have two 'the's, sometimes not
    words = line1.removesuffix(".").lower().split()
    unique_words = set(words)
    for word in unique_words:
        index = words.index(word)
        while word in words:
            words.remove(word)
        words.insert(index, word)
    line1 = generate.sentencify(" ".join(words))

    # If removing looping worked, no need to continue
    if well_formed(line1):
        return line1

    # Missing "is" - put back in
    if relation == "is" and " is " not in line1:
        words = line1.split()
        words.insert(-1, "is")
        reconstructed = " ".join(words)
        if well_formed(reconstructed):
            return reconstructed

    # Somewhat-recognisable alias for correct subject
    aliases = {
        "Anne": ["The anneagle"],
        "Bob": ["The bobster"],  # All hail the Bobster
        "Erin": ["The eragle", "The etah", "The eagle", "The er", "The ereagle"],
        "Fiona": ["The ficon", "The filion", "The fion"],
        "Harry": ["The h"],
        "the cheetah": ["Che"],
        "the crocodile": ["Cro"],
        "the squirrel": ["S"],
    }
    for subject, alias_list in aliases.items():
        for alias in alias_list:
            reconstructed = line1.replace(alias, subject)
            if well_formed(reconstructed):
                return generate.sentencify(reconstructed)

    # Missing 'the' - put back
    if "the" in line0.lower() and not "the" in line1.lower():
        attempt = generate.sentencify("the " + line1.lower())
        if well_formed(attempt):
            return attempt

    # Extra 'the' - remove
    if not "the" in line0.lower() and "the" in line1.lower():
        attempt = generate.sentencify(line1.lower().replace("the", "").strip())
        if well_formed(attempt):
            return attempt


def diagnose(line0: str, line1: str):
    subject, relation, attribute, obj = decompose(line0)

    if well_formed(line1):
        subject1, relation1, attribute1, object1 = decompose(line1)
        if [subject, relation, attribute, obj] == [
            subject1,
            relation1,
            attribute1,
            object1,
        ]:
            return "correct"
        if attribute == attribute1:
            return "salvageable"
        return "incorrect"

    fixed = attempt_to_fix(line0, line1)
    if fixed:
        return "fixable and " + diagnose(line0, fixed)

    if attribute in line1:
        for other_attribute in attributes:
            if other_attribute != attribute and other_attribute in line1:
                return "somewhat useful"
        return "useful" 
    if relation != "is":
        # Doesn't cover multiple relations or objects
        if relation[:-1] in line1 or obj in line1:
            return "somewhat useful"

    return "useless"  # Doesn't cover correct entity


def correct(line0, line1):
    return line0 == line1


def fixable(line0, line1):
    result = diagnose(line0, line1)
    return result == "correct" or result == "fixable and correct"


def useful(line0, line1):
    useful_results = [
        "correct",
        "salvageable",
        "fixable and correct",
        "fixable and salvageable",
        "useful",
    ]
    return diagnose(line0, line1) in useful_results


def classify_answers():
    for model in models:
        for dataset in datasets:
            classes = []
            results_file = os.path.join("results", dataset, f"results_{model}.txt")
            if os.path.exists(results_file):
                with open(results_file) as file:
                    contents = file.readlines()
                tuples = [ast.literal_eval(line) for line in contents]
                for line in tuples:
                    classes.append(diagnose(line[0], line[1]))
            report = Counter(classes)
            print(model, "on", dataset + ":", report)


def results_table(criterion=correct):
    table = [
        ["Model \\ Test set"]
        + [dataset.removeprefix("Abduction-") for dataset in datasets]
    ]
    for model in models:
        row = [model]
        for dataset in datasets:
            results_file = os.path.join("results", dataset, f"results_{model}.txt")
            if os.path.exists(results_file):
                with open(results_file) as file:
                    contents = file.readlines()
                tuples = [ast.literal_eval(line) for line in contents]
                results = [criterion(line[0], line[1]) for line in tuples]
                successes = results.count(True)
                failures = results.count(False)
                total = successes + failures
                row.append(as_percent(successes, total))
            else:
                row.append("")
        table.append(row)
    return table


def printable_table(rows):
    rows = [[str(item) for item in row] for row in rows]
    lengths = {}
    for row in rows:
        for i, item in enumerate(row):
            if i in lengths:
                lengths[i] = max(lengths[i], len(item))
            else:
                lengths[i] = len(item)
    new_rows = [[item.rjust(lengths[i]) for i, item in enumerate(row)] for row in rows]
    printable = "\n".join([" ".join(row) for row in new_rows])
    return printable


def latex_table(rows, caption=None, label=None):
    """Given a list of lists, converts into a LATEX table."""
    num_cols = max([len(row) for row in rows])

    string_list = ["\\begin{adjustbox}{max width=\\textwidth}\n"]
    string_list += ["\\begin{tabular}{|", "c|" * num_cols, "}\n\\hline\n"]

    for row in rows:
        for i in range(num_cols):
            if len(row) > i:
                if "%" in row[i]:
                    string_list.append("\\cellcolor{blue!")
                    string_list.append(str(float(row[i].removesuffix("%"))/2))
                    string_list.append("}")
                    if rows[0][i] in row[0].removeprefix("Abduction-").split("+"):
                        string_list.append("\\textbf{")
                string_list.append(str(row[i]))
                if "%" in row[i] and rows[0][i] in row[0].removeprefix("Abduction-").split("+"):
                        string_list.append("}")
            string_list.append("&")
        string_list[-1] = "\\\\\n\\hline\n"

    string_list.append("\\end{tabular}\n")
    string_list.append("\\end{adjustbox}")

    if caption != None:
        string_list.insert(0, "\\begin{table*}[t]\n")
        string_list.append("\n\\caption{")
        if label != None:
            string_list.append("\\label{tab:")
            string_list.append(str(label))
            string_list.append("}")
        string_list.append(str(caption))
        string_list.append("}\n\\end{table*}")

    string = "".join(string_list)
    string = string.replace("%", "\\%")
    string = string.replace(" \\ ", " $\\backslash$ ")

    return string


def main():
    criteria = [correct, fixable, useful]

    for criterion in criteria:
        category = criterion.__name__.capitalize()
        caption = category + " performance of all models on all test sets. Results from each model's associated test dataset(s) are bolded."
        label = category.lower() + "results"
        table = results_table(criterion)
        to_print = printable_table(table)
        # to_print = latex_table(table, caption, label)
        print(caption)
        print(to_print)
        print()


if __name__ == "__main__":
    main()
