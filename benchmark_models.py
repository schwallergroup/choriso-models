import eco2ai
from reaction_model import *


class BenchmarkPipeline:

    def __init__(self, model: ReactionModel):
        self.model = model
        # TODO Change the alpha_2_code to your country's IP, otherwise there may be issues with url timeouts
        self.alpha_2_code = "CH"

    def run_train_pipeline(self, dataset):
        tracker_path = f"{self.model.model_dir}/{dataset}/results"
        if not os.path.exists(tracker_path):
            os.makedirs(tracker_path)
        preprocess_tracker = eco2ai.Tracker(
            project_name=f"{self.model.name}_benchmark",
            experiment_description=f"preprocess {self.model.name} model",
            file_name=f"{tracker_path}/preprocess_emission.csv",
            alpha_2_code=self.alpha_2_code,
        )
        preprocess_tracker.start()
        self.model.preprocess(dataset=dataset)
        preprocess_tracker.stop()

        train_tracker = eco2ai.Tracker(
            project_name=f"{self.model.name}_benchmark",
            experiment_description=f"train {self.model.name} model",
            file_name=f"{tracker_path}/train_emission.csv",
            alpha_2_code=self.alpha_2_code,
        )
        train_tracker.start()
        self.model.train(dataset=dataset)
        train_tracker.stop()

    def predict(self, dataset):
        tracker_path = f"{self.model.model_dir}/{dataset}/results"
        if not os.path.exists(tracker_path):
            os.makedirs(tracker_path)
        predict_tracker = eco2ai.Tracker(
            project_name=f"{self.model.name}_benchmark",
            experiment_description=f"predict {self.model.name} model",
            file_name=f"{tracker_path}/predict_emission.csv",
            alpha_2_code=self.alpha_2_code,
        )
        predict_tracker.start()
        self.model.predict(dataset=dataset)
        predict_tracker.stop()

    def run_pipeline(self, dataset):
        self.run_train_pipeline(dataset=dataset)
        self.predict(dataset=dataset)

    def run_mode_from_args(self, args):
        if args.phase == "t":
            self.run_train_pipeline(dataset=args.dataset)

        elif args.phase == "p":
            self.predict(dataset=args.dataset)

        elif args.phase == "tp":
            self.run_pipeline(dataset=args.dataset)

        else:
            raise ValueError(f"Unknown mode {args.mode}")
