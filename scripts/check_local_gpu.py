import os
import platform
import subprocess
import sys


def run(command):
    print(f"$ {' '.join(command)}")
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        print("not found")
        return
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip())
    print(f"exit={result.returncode}")


def main():
    print("Python:", sys.version.replace("\n", " "))
    print("Platform:", platform.platform())
    print("WSL_DISTRO_NAME:", os.environ.get("WSL_DISTRO_NAME"))
    print()

    run(["nvidia-smi"])
    print()

    try:
        import torch
    except Exception as exc:
        print("torch import failed:", repr(exc))
        return

    print("torch:", torch.__version__)
    print("torch.version.cuda:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device count:", torch.cuda.device_count())
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            print(
                f"cuda:{index}",
                props.name,
                f"{props.total_memory / (1024 ** 3):.1f} GiB",
            )


if __name__ == "__main__":
    main()
