import json
import os
import random
import shutil

import torch
from transformers import Adafactor, T5ForConditionalGeneration, T5Tokenizer

from analysis import as_percent
from analysis import main as print_results
from generate import datasets as dataset_names
from generate import main as generate_datasets

datasets = {}
models = dataset_names.copy()


def sample_two_lists(list1, list2, k):
    return zip(*random.sample(list(zip(list1, list2)), k))


def combine_datasets(*indices):
    name = "+".join([dataset_names[i].removeprefix("Abduction-") for i in indices])
    random.seed(name)
    partitions = ["dev", "train", "test"]
    combined_data = {}
    for part in partitions:
        combined_data[part] = {"inputs": [], "outputs": []}
        for i in indices:
            dataset = datasets[dataset_names[i]]
            combined_data[part]["inputs"].extend(
                dataset[part]["inputs"])
            combined_data[part]["outputs"].extend(
                dataset[part]["outputs"])
        length = len(combined_data[part]["inputs"])
        combined_data[part]["inputs"], combined_data[part]["outputs"] = sample_two_lists(
            combined_data[part]["inputs"], combined_data[part]["outputs"], length)
    datasets[name] = combined_data
    models.append(name)


def generate(text, model, tokenizer, device):
    model.eval()
    input_ids = tokenizer.encode(text, return_tensors="pt")
    input_ids = input_ids.to(device)
    outputs = model.generate(input_ids)
    return tokenizer.decode(outputs[0])


def answer_question(context, observation):
    model = T5ForConditionalGeneration.from_pretrained(os.path.curdir)
    text = context + "\n" + observation.removeprefix(".") + "?"
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    explanation = generate(text, model, tokenizer, get_device())
    return explanation[6:-4]


def add_pararules(folder):
    if folder in datasets:
        return

    partitions = ["train", "dev", "test"]
    data = {}
    nl = "\n"

    for part in partitions:
        data[part] = {"inputs": [], "outputs": []}
        with open(os.path.join("datasets", folder, part + ".jsonl")) as file:
            raw = [json.loads(line) for line in file.readlines()]
        for item in raw:
            for question in item["questions"]:
                data[part]["inputs"].append(
                    f'{item["context"]}\n{question["text"][:-1]}?'
                )
                data[part]["outputs"].append(question["label"])

    datasets[folder] = data


def get_model(folder=None, from_scratch=False):
    if folder == None:
        if from_scratch:
            return T5ForConditionalGeneration(return_dict=True)
        return T5ForConditionalGeneration.from_pretrained(
            "t5-base", return_dict=True)
    if not os.path.exists(os.path.join("models", folder, "config.json")):
        shutil.copyfile("config.json", os.path.join(
            "models", folder, "config.json"))
    return T5ForConditionalGeneration.from_pretrained(
        os.path.join("models", folder), return_dict=True)


def get_device():
    device_num = 0
    if torch.cuda.is_available():
        dev = torch.device(f"cuda:{device_num}")
        print(f"Running on GPU no. {device_num}")
    else:
        dev = torch.device("cpu")
        print("Running on CPU")

    return dev


def get_data(folder, set, test=False):
    dataset = datasets[folder][set]
    inputs = dataset["inputs"]
    labels = dataset["outputs"]
    if test:
        inputs, labels = sample_two_lists(inputs, labels, 10)

    return inputs, labels


def train_model(folder, from_scratch=False, test=False):
    model_location = os.path.join("models", folder, "pytorch_model.bin")
    if os.path.exists(model_location):
        print(f"{folder} model already exists, skipping training")
        return

    print(f"Training model on {folder} set(s)")
    model = get_model(from_scratch=from_scratch)
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    optimizer = Adafactor(
        model.parameters(),
        lr=1e-3,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
    )

    inputs, labels = get_data(folder, "train", test)

    batch_size = 5
    batches = len(inputs) // batch_size

    dev = get_device()
    model.to(dev)
    model.train()

    epochs = 10
    for epoch in range(1, epochs + 1):
        print(f"Running {epoch=}")

        running_loss = 0

        for batch in range(batches):
            inputbatch = inputs[batch * batch_size: (batch+1) * batch_size]
            labelbatch = labels[batch * batch_size: (batch+1) * batch_size]
            inputbatch = tokenizer.batch_encode_plus(
                inputbatch, padding=True, max_length=400, return_tensors="pt"
            )["input_ids"]
            labelbatch = tokenizer.batch_encode_plus(
                labelbatch, padding=True, max_length=400, return_tensors="pt"
            )["input_ids"]
            inputbatch = inputbatch.to(dev)
            labelbatch = labelbatch.to(dev)

            optimizer.zero_grad()
            outputs = model(input_ids=inputbatch, labels=labelbatch)
            loss = outputs.loss
            loss_num = loss.item()
            running_loss += loss_num

            if batch % 100 == 0:
                print(
                    f"{batch=} of {batches}; {epoch=} of {epochs}; {as_percent(epoch-1+batch/batches, epochs)} done")

            loss.backward()
            optimizer.step()

        running_loss = running_loss / int(batches)
        print(f"Finished epoch {epoch}: {running_loss=}")

    if not os.path.exists("models"):
        os.mkdir("models")
    if not os.path.exists(os.path.join("models", folder)):
        os.mkdir(os.path.join("models", folder))
    torch.save(model.state_dict(), model_location)


def test_model(model_folder, test_folder, test=False):
    results_file = os.path.join(
        "results", test_folder, f"results_{model_folder}.txt")
    if os.path.exists(results_file):
        print(f"{model_folder} model already tested on {test_folder} set, skipping")
        return

    if not os.path.exists(os.path.join("models", model_folder, "pytorch_model.bin")):
        print(f"{model_folder} model doesn't exist!")
        train_model(model_folder, test=test)
        print(f"{model_folder} model trained")

    print(f"Testing {model_folder} model on {test_folder} set")

    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = get_model(model_folder)
    dev = get_device()
    model.to(dev)

    inputs, labels = get_data(test_folder, "test", test)

    results = []
    total = 0
    successes = 0

    length = len(inputs)
    for i in range(length):
        question = inputs[i]
        label = labels[i]
        output = generate(question, model, tokenizer, dev)[6:-4]
        success = label == output
        result = (label, output, success)
        results.append(str(result) + "\n")
        total += 1
        successes += success
        if i % 100 == 1:
            print(
                f"{as_percent(i, length)} done; {as_percent(successes, total)} accuracy")

    if not os.path.exists("results"):
        os.mkdir("results")
    if not os.path.exists(os.path.join("results", test_folder)):
        os.mkdir(os.path.join("results", test_folder))
    with open(results_file, "w") as file:
        file.writelines(results)

    print(
        f"FINAL RESULTS: {successes} correct out of {total} ({as_percent(successes, total)})")


def main():
    generate_datasets()
    test = False
    for dataset in dataset_names:
        add_pararules(dataset)
    combine_datasets(3, 4)  # Animal+Person-Simple
    combine_datasets(5, 0)  # Person+Animal-0.1
    for model in models:
        for dataset in dataset_names:
            test_model(model, dataset, test)
    print_results()


if __name__ == "__main__":
    main()
