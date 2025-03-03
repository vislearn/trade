import sys
import subprocess
import json
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--max_epochs", type=int)
    parser.add_argument("--limit_train_batches", type=int)
    parser.add_argument("--limit_val_batches", type=int)
    parser.add_argument("--limit_test_batches", type=int)
    parser.add_argument("--fast_dev_run", type=bool)
    parser.add_argument("--overfit_batches", type=int)
    args, unknown = parser.parse_known_args()

    # Convert arguments to dictionary
    trainer_kwargs = {
        k: v for k, v in vars(args).items() if v is not None and k not in ["config"]    
}

    # Convert to JSON string and construct new command
    command = [
        sys.executable, "-m", "lightning_trainable.launcher.fit",
        args.config, f"--trainer-kwargs={json.dumps(trainer_kwargs)}"
    ] + unknown  # Preserve additional arguments

    # Run the actual module
    subprocess.run(command)

if __name__ == "__main__":
    main()
