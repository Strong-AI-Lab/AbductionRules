import os


def as_percent(numerator, denominator, ndigits=1):
    return str(round(numerator/denominator*100, ndigits=ndigits)) + "%"


all_datasets = ["Abduction-Animal-0.1", "Abduction-Animal-0.2",
                "Abduction-Animal-Simple", "Abduction-Animal", "Abduction-Person-Simple", "Abduction-Person"]
all_models = ["Abduction-Animal-0.1", "Abduction-Animal-0.2",
              "Abduction-Animal-Simple", "Abduction-Animal", "Abduction-Person-Simple", "Abduction-Person", "Animal+Person-Simple"]


def main():
    for model in all_models:
        for dataset in all_datasets:
            results_file = os.path.join(
                "results", dataset, f"results_{model}.txt")
            if os.path.exists(results_file):
                with open(results_file) as file:
                    contents = file.read()
                trues = contents.count("True")
                falses = contents.count("False")
                total = trues + falses
                print(
                    f"{model} model performance on {dataset} dataset: {trues} out of {total}, which is {as_percent(trues, total)}")


if __name__ == "__main__":
    main()
