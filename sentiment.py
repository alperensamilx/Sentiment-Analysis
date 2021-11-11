from transformers import pipeline

import torch
import torch.nn.functional as F


def freeze_support():
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"


    classifier = pipeline("sentiment-analysis", model=model_name)
    results = classifier(["We are very happy to show you the 🤗 Transformers library.",
                          "We hope you don't hate it."])

    for result in results:
        print(result)


if __name__ == '__main__':
    freeze_support()