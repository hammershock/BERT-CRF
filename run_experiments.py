import os
import subprocess


def run_experiments():
    num_hidden_layers_list = [4, 6, 8, 10]
    default_args = {
        "num_epochs": 10,
        "batch_size": 420,
        "lr": 5e-5,
        "num_labels": 9,
        "save_dir": "./models",
        "save_every": 1,
    }

    for num_hidden_layers in num_hidden_layers_list:
        args = default_args.copy()
        args["num_hidden_layers"] = num_hidden_layers

        # Construct the command
        command = [
            "python", "main.py",
            f"--num_epochs={args['num_epochs']}",
            f"--batch_size={args['batch_size']}",
            f"--lr={args['lr']}",
            f"--num_labels={args['num_labels']}",
            f"--num_hidden_layers={args['num_hidden_layers']}",
            f"--save_dir={args['save_dir']}",
            f"--save_every={args['save_every']}"
        ]

        # Print the command to be executed
        print("Running command:", " ".join(command))

        # Execute the command
        subprocess.run(command)


if __name__ == "__main__":
    run_experiments()
