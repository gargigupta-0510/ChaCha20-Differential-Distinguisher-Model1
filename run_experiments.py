import subprocess
import re
import os
import sys


rounds_to_test = [
    3,
    4,
    5,
    6,
    7,
    8
]


results = []


for rounds in rounds_to_test:
    print("\n")
    print("=" * 60)
    print(f"RUNNING {rounds}-ROUND CHACHA")
    print("=" * 60)


    # Modify ROUNDS in dataset generator
    with open("generate_dataset.py", "r", encoding="utf-8") as f:
        code = f.read()


    code = re.sub(
        r"ROUNDS = \d+",
        f"ROUNDS = {rounds}",
        code
    )


    with open("generate_dataset.py", "w", encoding="utf-8") as f:
        f.write(code)


    # Generate dataset
    subprocess.run(
        [sys.executable, "generate_dataset.py"],
        check=True
    )


    # Train
    subprocess.run(
        [sys.executable, "train.py"],
        check=True
    )


    # Evaluate
    result = subprocess.run(
        [sys.executable, "evaluate.py"],
        capture_output=True,
        text=True,
        check=True
    )


    print(result.stdout)


    # Extract accuracy
    match = re.search(r"Accuracy\s+:\s+([0-9.]+)", result.stdout)

    if match:
        accuracy = float(match.group(1))
        results.append((rounds, accuracy))


print("\n\nFINAL RESULTS")
print("========================")

for rounds, accuracy in results:
    print(f"{rounds} rounds : "f"{accuracy:.4f}")