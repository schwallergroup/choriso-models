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

    def run_mode_from_args(self, args):
        if args.mode == "t":
            self.run_train_pipeline(dataset=args.dataset)

        elif args.mode == "p":
            self.predict(dataset=args.dataset)

        elif args.mode == "tp":
            self.run_pipeline(dataset=args.dataset)

        else:
            raise ValueError(f"Unknown mode {args.mode}")
