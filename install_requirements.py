import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def main():
    with open("requirements.txt") as requirements_file:
        for line in requirements_file:
            package = line.strip()
            install(package)

if __name__ == "__main__":
    main()

