from reaction_model import *


class BenchmarkPipeline:

    def __init__(self, model: ReactionModel):
        self.model = model

    def run_train_pipeline(self, dataset):
        self.model.preprocess(dataset=dataset)
        self.model.train(dataset=dataset)

    def predict(self, dataset):
        self.model.predict(dataset=dataset)

    def run_pipeline(self, dataset):
        self.model.preprocess(dataset=dataset)
        self.model.train(dataset=dataset)
        self.model.predict(dataset=dataset)
