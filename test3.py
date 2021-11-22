import torch
import pandas as pd
import numpy as np
from wilds.datasets.iwildcam_dataset import IWildCamDataset
from wilds.datasets.poverty_dataset import PovertyMapDataset
from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
from wilds.datasets.fmow_dataset import FMoWDataset

root = "./data/"
dataset = FMoWDataset(root_dir=root)

#root = "../DomainBed/domainbed/data/poverty_v1.1/"
#dataset = PovertyMapDataset(root_dir=root)

#root= "../DomainBed/domainbed/data/camelyon17_v1.0/"
#dataset = Camelyon17Dataset(root_dir=root)

def metadata_values(wilds_dataset, metadata_name):
    metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
    metadata_vals = wilds_dataset.metadata_array[:, metadata_index]
    return sorted(list(set(metadata_vals.view(-1).tolist())))

#print(dir(dataset))
#print(dataset.metadata_fields)
#print(dataset.metadata_map)
#print(dataset.metadata_array)
domain = "region"
unique_domains = metadata_values(dataset, domain)
print(unique_domains)
print(dataset.metadata_fields)
print(dataset.metadata_array)

df = pd.DataFrame(dataset.metadata_array.numpy(), columns = dataset.metadata_fields)
#print(df)
sample_list = []
classes_list=[]
for i in range(len(unique_domains)):
    num_samples = df[df[domain] == unique_domains[i]]
    print(len(num_samples))
    y_vals = num_samples["y"]
    num_classes = np.unique(y_vals)
    for nc in num_classes:
        samples_in_that_class = num_samples[num_samples["y"] == nc]
        print(f"location: {unique_domains[i]}, class= {nc}, num samples: {len(samples_in_that_class)}")
    #sample_list.append(len(num_samples))
    #classes_list.append(len(num_classes))
    # print(f"location: {unique_domains[i]}, num samples: {len(num_samples)}, num classes: {len(num_classes)}")

#new_df = pd.DataFrame(list(zip(sample_list,classes_list)),columns = ["#samples", "#classes"])
#new_df.to_csv("sample-classes.csv")
# envs = []
# for i, dom in enumerate(unique_domains):
#     envs.append("country_" + str(dom))
#     #print("country_" + str(dom))
# print(envs)
