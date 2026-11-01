import torch


class OptimizerFactory:
    @staticmethod
    def create(name, model_parameters, learning_rate, momentum=0.9, weight_decay=0.0):
        name = name.lower()

        if name == "sgd":
            return torch.optim.SGD(model_parameters, lr=learning_rate, weight_decay=weight_decay)

        elif name == "nesterov_sgd":
            return torch.optim.SGD(
                model_parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay, nesterov=True
            )

        elif name == "adam":
            return torch.optim.Adam(model_parameters, lr=learning_rate, weight_decay=weight_decay)

        elif name == "adagrad":
            return torch.optim.Adagrad(model_parameters, lr=learning_rate, weight_decay=weight_decay)

        elif name == "rmsprop":
            return torch.optim.RMSprop(
                model_parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay
            )

        elif name == "nadam":
            return torch.optim.NAdam(model_parameters, lr=learning_rate, weight_decay=weight_decay)

        else:
            raise ValueError(f"Unsupported optimizer: '{name}'")