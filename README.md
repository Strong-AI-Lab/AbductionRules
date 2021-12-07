# AbductionRules
A collection of synthetic natural-language logic datasets, designed to train and test abductive reasoning and modeled on the [Rule Reasoning dataset](https://allenai.org/data/ruletaker) and [PARARULE Plus](https://github.com/Strong-AI-Lab/PARARULE-Plus).

These datasets form the basis of [AbductionRules: Training Transformers to Explain Unexpected Inputs](https://github.com/Strong-AI-Lab/AbductionRules/blob/main/abductionrules_paper.pdf).

Each dataset item contains a collection of facts and rules (the 'context') along with 8 or 10 observations, each of which may be proven or disproven using the context and one additional fact.

In addition to training a different kind of reasoning, AbductionRules iterates on the natural-language logic paradigm by eschewing premade templates in favour of procedural rephrasing of rules, aloowing the same rule to be phrased as "If something is big, is heavy, and is fierce, it is strong." or "All animals that are fierce, are big, and are heavy, are also strong."

## Datasets

AbductionRules contains 4 main datasets, spanning 2 domains and 2 levels of complexity:

|Domain \\ Complexity|Simple|Confounded|
|:---|:---|:---|
|Animal|Abduction-Animal-Simple|Abduction-Animal|
|Person|Abduction-Person-Simple|Abduction-Person|

The domains differ mainly in their entities ("the dog" vs "Charlie") and attributes ("furry" vs. "wealthy").
The animal-domain also incorporates multi-entity rules, e.g. "the lion chases the mouse".

Simple datasets contain only one rule that can be used to explain an observation, yielding only one possible answer.

To these we added confounding rules that can produce more complex explanations for the same observation; these are to be ignored by the model in favour of inference to the simplest explanation.

(Datasets with lower complexity, produced without our rephrasing method, may be generated with generate.py in the animal-domain only. These datasets, Abduction-Animal-0.1 and -0.2, are not available in the public release of AbductionRules due to their inferiority to Abduction-Animal-Simple.)

## Replicating Experiments

To replicate the experiments in the [paper](https://github.com/Strong-AI-Lab/AbductionRules/blob/main/abductionrules_paper.pdf), simply run main.py.
This requires the [HuggingFace transformers library](https://github.com/huggingface/transformers), which may be installed with `pip install transformers`.