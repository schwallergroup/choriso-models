import torch
from torch.utils.data import Dataset


class TemplateData(Dataset):

    def __init__(self, fps, template_idx):

        self.fps = fps
        self.template_idx = template_idx

    def __len__(self):
        return len(self.template_idx)

    def __getitem__(self, idx):
        return self.fps[idx], self.template_idx[idx]


class TemplateClassifier(torch.nn.Module):

    def __init__(self, fp_size=1024, num_classes=8720):
        super().__init__()

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(fp_size, 512),
            torch.nn.ELU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(512, num_classes),
            torch.nn.Softmax(dim=-1)
        )

    def forward(self, data):
        return self.mlp(data)
