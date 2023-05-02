import eco2ai
from reaction_model import *


class BenchmarkPipeline:

    def __init__(self, model: ReactionModel):
        self.model = model

    def run_train_pipeline(self, dataset):
        preprocess_tracker = eco2ai.Tracker(
            project_name=f"{self.model.name}_benchmark",
            experiment_description=f"preprocess {self.model.name} model",
            file_name=f"{self.model.model_dir}/{dataset}/preprocess_emission.csv"
        )
        preprocess_tracker.start()
        self.model.preprocess(dataset=dataset)
        preprocess_tracker.stop()

        train_tracker = eco2ai.Tracker(
            project_name=f"{self.model.name}_benchmark",
            experiment_description=f"train {self.model.name} model",
            file_name=f"{self.model.model_dir}/{dataset}/train_emission.csv"
        )
        train_tracker.start()
        self.model.train(dataset=dataset)
        train_tracker.stop()

    def predict(self, dataset):
        predict_tracker = eco2ai.Tracker(
            project_name=f"{self.model.name}_benchmark",
            experiment_description=f"predict {self.model.name} model",
            file_name=f"{self.model.model_dir}/{dataset}/predict_emission.csv"
        )
        predict_tracker.start()
        self.model.predict(dataset=dataset)
        predict_tracker.stop()

    def run_pipeline(self, dataset):
        self.run_train_pipeline(dataset=dataset)
        self.predict(dataset=dataset)

    def run_mode_from_args(self, args):
        if args.mode == "t":
            self.run_train_pipeline(dataset=args.dataset)

        elif args.mode == "p":
            self.predict(dataset=args.dataset)

        elif args.mode == "tp":
            self.run_pipeline(dataset=args.dataset)

        else:
            raise ValueError(f"Unknown mode {args.mode}")
