import torch


class AdataDataset(torch.utils.data.Dataset):
    def __init__(self, x, covariates):
        super().__init__()
        self.x = x
        self.covariates = covariates

    def __getitem__(self, index):
        return self.x[index], self.covariates[index]

    def __len__(self):
        return len(self.x)
