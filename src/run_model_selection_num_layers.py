import subprocess
from itertools import product


def run_experiments():
    num_hidden_layers_list = [10, 8, 6, 4]
    pretrained_list = [0, 1]
    model_type = "bert_crf"

    for num_hidden_layers, pretrained in product(num_hidden_layers_list, pretrained_list):
        log_path = f"./logs/layers{num_hidden_layers}_pretrained{pretrained}.log"
        command = [
            "python", "train.py",
            f"--num_hidden_layers={num_hidden_layers}",
            f"--log_path={log_path}",
            f"--pretrained={pretrained}",
            f"--model_type={model_type}",
            f"--lr={5e-5 if pretrained == 2 else 5e-4}",
        ]

        # Print the command to be executed
        print("Running command:", " ".join(command))

        # Execute the command
        subprocess.run(command)


if __name__ == "__main__":
    run_experiments()
