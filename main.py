import itertools
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
models = dataset_names + ["Zero-Shot"]


def sample_two_lists(list1, list2, k):
    return zip(*random.sample(list(zip(list1, list2)), k))


def combine_datasets(*indices):
    partitions = ["train", "dev", "test"]
    parts = [dataset_names[index] for index in indices]
    name = "+".join([part.replace("Abduction-", "") for part in parts])
    models.append(name)
    random.seed(name)
    datasets[name] = {}
    for partition in partitions:
        datasets[name][partition] = list(
            itertools.chain.from_iterable([datasets[part][partition] for part in parts])
        )
        random.shuffle(datasets[name][partition])


def generate(text, model, tokenizer, device):
    model.eval()
    input_ids = tokenizer.encode(text, return_tensors="pt")
    input_ids = input_ids.to(device)
    outputs = model.generate(input_ids)
    return as_output(tokenizer.decode(outputs[0]))


def as_input(context, observation):
    return context + "\n" + observation.removesuffix(".") + "?"


def as_output(raw: str):
    while "<" in raw:
        beginning = raw.find("<")
        end = raw.find(">")
        raw = raw[:beginning] + raw[end + 1 :]
    return raw.strip()


def answer_question(context, observation):
    model = T5ForConditionalGeneration.from_pretrained(os.path.curdir)
    query = as_input(context, observation)
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    explanation = generate(query, model, tokenizer, get_device())
    return explanation


def add_dataset(folder):
    if folder in datasets:
        return

    partitions = ["train", "dev", "test"]
    data = {}

    for part in partitions:
        with open(os.path.join("datasets", folder, part + ".jsonl")) as file:
            data[part] = [json.loads(line) for line in file.readlines()]

    datasets[folder] = data


def get_model(folder=None, from_scratch=False):
    if folder == None:
        if from_scratch:
            return T5ForConditionalGeneration(return_dict=True)
        return T5ForConditionalGeneration.from_pretrained("t5-base", return_dict=True)
    if not os.path.exists(os.path.join("models", folder, "config.json")):
        shutil.copyfile("config.json", os.path.join("models", folder, "config.json"))
    return T5ForConditionalGeneration.from_pretrained(
        os.path.join("models", folder), return_dict=True
    )


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
    inputs = []
    labels = []
    dataset = datasets[folder][set]
    for item in dataset:
        for question in item["questions"]:
            inputs.append(as_input(item["context"], question["text"]))
            labels.append(question["label"])
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
            inputbatch = inputs[batch * batch_size : (batch + 1) * batch_size]
            labelbatch = labels[batch * batch_size : (batch + 1) * batch_size]
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
                    f"{batch=} of {batches}; {epoch=} of {epochs}; {as_percent(epoch-1+batch/batches, epochs)} done"
                )

            loss.backward()
            optimizer.step()

        running_loss = running_loss / int(batches)
        print(f"Finished epoch {epoch}: {running_loss=}")

    if not os.path.exists("models"):
        os.mkdir("models")
    if not os.path.exists(os.path.join("models", folder)):
        os.mkdir(os.path.join("models", folder))
    torch.save(model.state_dict(), model_location)


def test_model(model_name, test_set, test=False):
    results_file = os.path.join("results", test_set, f"results_{model_name}.jsonl")
    if os.path.exists(results_file):
        print(f"{model_name} model already tested on {test_set} set, skipping")
        return

    if model_name == "Zero-Shot":
        model = T5ForConditionalGeneration.from_pretrained("t5-base", return_dict=True)
    else:
        if not os.path.exists(os.path.join("models", model_name, "pytorch_model.bin")):
            print(f"{model_name} model doesn't exist!")
            train_model(model_name, test=test)
            print(f"{model_name} model trained")
        model = get_model(model_name)

    print(f"Testing {model_name} model on {test_set} set")

    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    dev = get_device()
    model.to(dev)

    dataset = datasets[test_set]["test"]
    if test:
        dataset = random.sample(dataset, 1)

    results = []
    total = 0
    successes = 0

    for item in dataset:
        for question in item["questions"]:
            query = as_input(item["context"], question["text"])
            output = generate(query, model, tokenizer, dev)
            success = question["label"] == output
            result = {
                "id": question["id"],
                "label": question["label"],
                "answer": output,
                "success": success,
            }
            results.append(result)
            total += 1
            successes += success
            if total % 100 == 1:
                print(
                    f"{as_percent(total, len(dataset)*len(item['questions']))} done; {as_percent(successes, total)} accuracy"
                )

    if not os.path.exists("results"):
        os.mkdir("results")
    if not os.path.exists(os.path.join("results", test_set)):
        os.mkdir(os.path.join("results", test_set))
    with open(results_file, "w") as file:
        for result in results:
            json.dump(result, file)
            file.write("\n")

    print(
        f"FINAL RESULTS: {successes} correct out of {total} ({as_percent(successes, total)})"
    )


def main():
    generate_datasets()
    test = True
    for dataset in dataset_names:
        add_dataset(dataset)
    combine_datasets(3, 4)  # Animal+Person-Simple
    combine_datasets(5, 0)  # Person+Animal-0.1
    for model in models:
        for dataset in dataset_names:
            test_model(model, dataset, test)
    print_results()


if __name__ == "__main__":
    main()
