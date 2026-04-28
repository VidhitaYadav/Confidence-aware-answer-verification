"""
Optional training placeholder for the CAAVS project.

The deployed app does not depend on this file anymore.
Inference now uses pretrained QA + NLI models directly because they are
more reliable than the previous broken custom checkpoint.

You can keep this file in the project so your submission still has a
training component, but the demo and deployment use app.py only.
"""


def main():
    print("This project now uses pretrained QA + NLI models for deployment.")
    print("No custom training step is required to run the app.")
    print("If you want, you can later replace this file with a proper fine-tuning pipeline.")


if __name__ == "__main__":
    main()
