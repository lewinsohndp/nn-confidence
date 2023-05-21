import torch
from datasets import load_dataset
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torchvision import transforms
import seaborn as sns
from reliability_diagrams import *
import os
import pandas as pd

def test_model(model_path, model_version, dataset_name, split = 'validation', n=5000, seed=8, topn=5):
    """method that returns model predictions, probabilities and real labels"""
    torch.manual_seed(seed)
    model = torch.hub.load(model_path, model_version, pretrained=True)
    model.eval()
    dataset = load_dataset(dataset_name, split=split, streaming=True)

    top_preds = np.zeros((n, topn))
    top_probs = np.zeros((n, topn))
    real_labels = np.zeros(n)

    bad_data_flag = False

    i = 0
    for data in dataset:
        input_image = data['image']
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        try:
            input_tensor = preprocess(input_image)
        except RuntimeError:
            bad_data_flag = True
            continue

        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

        with torch.no_grad():
            output = model(input_batch)

        # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        topn_prob, topn_catid = torch.topk(probabilities, topn)

        top_preds[i] = topn_catid
        top_probs[i] = topn_prob
        real_labels[i] = data['label']
        if i%10000 == 0: print(i)
        if i == n-1: break
        i+=1
    if bad_data_flag: print('Data likely has bad inputs, these were skipped')
    return top_preds, top_probs, real_labels

def main():
    models = ['shufflenet_v2_x1_0', 'densenet121', 'resnet18']
    n = 50000
    for model in models:
        print('starting model: ' + model)

        model_path = 'pytorch/vision:v0.10.0'
        model_version = model
        dataset_name = 'imagenet-1k'
        preds, probs, labels = test_model(model_path, model_version, dataset_name, n=n, topn=10)

        newpath = model 
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        preds_df = pd.DataFrame(preds)
        probs_df = pd.DataFrame(probs)
        labels_df = pd.DataFrame(labels)

        preds_df.to_csv(newpath + "/preds.csv")
        probs_df.to_csv(newpath + "/probs.csv")
        labels_df.to_csv(newpath + "/labels.csv")

if __name__ == "__main__":
    main()