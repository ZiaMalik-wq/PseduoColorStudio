# Color Studio

Color Studio is a Python project for color processing that includes a convolutional neural network (CNN) model and supporting algorithms for mapping, slicing, and LUT processing.

## Project Structure

- **app.py**: Application orchestrator.
- **main.py**: Primary entry point.
- **algorithms/**: Core processing modules (cnn, lut, mapping, slicing).
- **ui/**: Modular UI components.
- **models/**: Saved model weights used by the CNN.
- **images/**: Sample images for testing.
- **notebooks/**: Research and training notebooks (including training.ipynb).
- **tests/**: Unit tests for algorithm validation.

## Running the Project

See **ReadMe.txt** for step-by-step setup and run commands.

## Training the Model

1. Open **notebooks/training.ipynb** (e.g., in Kaggle or Jupyter) and run all cells to train the model.
2. Trained weights are saved in the **models/** directory.
