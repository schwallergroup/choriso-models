import os
import torch
from torch.utils.data import DataLoader

from reaction_model import ReactionModel, ReactionModelArgs
from TemplateModel.preprocess import process as preprocess_template
from TemplateModel.model import TemplateClassifier, TemplateData


class TemplateModel(ReactionModel):

    def __init__(self):
        self.name = "TemplateModel"
        super().__init__()

    def preprocess(self, dataset="cjhif"):
        """Do data preprocessing. Skip if preprocessed data already exists"""
        data_dir = os.path.join(os.path.dirname(self.model_dir), "data", dataset)
        preprocess_template(data_dir=data_dir)

    def train(self, dataset="cjhif"):
        epochs = 30
        lr_scheduler_factor = 0.3
        lr_scheduler_patience = 1
        lr_cooldown = 0
        batch_size = 256

        data_dir = os.path.join(os.path.dirname(self.model_dir), "data", dataset)
        data_dict = {}
        num_templates = 0

        for data_split in ["train", "valid", "test"]:
            fps = torch.load(os.path.join(data_dir, f"{data_split}_fps.pt"))
            template_idx = torch.load(os.path.join(data_dir, f"{data_split}_fps.pt"))
            num_templates = max(num_templates, max(template_idx))
            data_dict[data_split] = TemplateData(fps, template_idx)

        fp_size = fps[0].shape[0]

        train_loader = DataLoader(data_dict["train"], batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(data_dict["valid"], batch_size=batch_size, shuffle=True)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = TemplateClassifier(fp_size, num_templates)
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters())

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode='max',  # monitor top-1 val accuracy
            factor=lr_scheduler_factor,
            patience=lr_scheduler_patience,
            cooldown=lr_cooldown,
            verbose=True
        )

        for epoch in range(epochs):
            model.train()
            for batch_idx, batch in enumerate(train_loader):
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch)
                loss = torch.nn.functional.cross_entropy(out, batch.y)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                for batch_idx, batch in enumerate(valid_loader):
                    batch = batch.to(device)
                    out = model(batch)
                    loss = torch.nn.functional.cross_entropy(out, batch.y)
                    _, pred = out.max(dim=1)
                    acc = pred.eq(batch.y).sum().item() / batch.y.shape[0]
                    print(f"Epoch {epoch}, batch {batch_idx}, loss {loss}, acc {acc}")

            lr_scheduler.step(acc)

    def predict(self, dataset="cjhif"):
        batch_size = 256

        data_dir = os.path.join(os.path.dirname(self.model_dir), "data", dataset)
        fps = os.path.join(data_dir, "test_fps.pt")
        template_idx = os.path.join(data_dir, "test_fps.pt")
        test_data = TemplateData(fps, template_idx)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

