from dataclasses import dataclass
import yaml
from monotone.functional import get_normed_weights
import json
import os 

@dataclass
class ModelConfiguration:
    """
    Object used to read model configuration data, given a .yml configuration file as an input.
    """

    infile: str
    data: dict = None
    model: str = None
    features: list = None
    def __post_init__(self) -> None:
        with open(self.infile, "r") as f:
            self.data = yaml.full_load(f)
        self.train_size = self.data['training_parameters']['train_size']
        self.features = self.data['features']
        self.shuffle = self.data['training_parameters']['shuffle']
        self.monotone_constraints=self.data['arch_parameters']['monotone_constraints']
        self.n_features = self.data['arch_parameters']['n_features']
        self.group_size = self.data['arch_parameters']['group_size']
        self.n_nodes = self.data['arch_parameters']['n_nodes']
        self.n_classes = self.data['arch_parameters']['n_classes']
        self.lambda_lip = self.data['arch_parameters']['lambda_lip']

        


def save_model_json(training_directory,model):
    state_dict = model.state_dict()
    weight_keys = [x for x in state_dict if "weight" in x]
    for k in state_dict:
        if k in weight_keys:
            state_dict[k] = get_normed_weights(
                state_dict[k],
                # ordered dict -> last weight_key is the one that
                # needs to be one-inf normed
                "one",
                always_norm= False,
                alpha= None,
                vectorwise= True
            ).tolist()
        else:
            state_dict[k] = state_dict[k].tolist()

    with open(os.path.join(training_directory,"normed_weights.json"), "w") as outfile:
        json.dump(state_dict, outfile)
        
