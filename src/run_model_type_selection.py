import subprocess
from itertools import product


def run_experiments():
    model_types_list = ["bert_crf", "bert_softmax"]
    pretrained_list = [0, 1, 2]
    num_hidden_layers = 12
    # pretrained

    for model_type, pretrained in product(model_types_list, pretrained_list):
        log_path = f"./logs/model_type{model_type}_pretrained{pretrained}.log"
        command = [
            "python", "train.py",
            f"--model_type={model_type}",
            f"--log_path={log_path}",
            f"--pretrained={pretrained}",
            f"--num_hidden_layers={num_hidden_layers}",
            f"--lr={5e-5 if pretrained == 2 else 5e-4}",
        ]

        # Print the command to be executed
        print("Running command:", " ".join(command))

        # Execute the command
        subprocess.run(command)


if __name__ == "__main__":
    run_experiments()
