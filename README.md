# Color Studio

Color Studio is a Python project for color processing that includes a convolutional neural network (CNN) model (trained on the **STL-10 dataset**) and supporting algorithms for mapping, slicing, and LUT processing.

## Project Structure

- **app.py**: Application orchestrator.
- **main.py**: Primary entry point.
- **algorithms/**: Core processing modules (cnn, lut, mapping, slicing).
- **ui/**: Modular UI components.
- **models/**: Saved model weights used by the CNN.
- **images/**: Sample images for testing.
- **notebooks/**: Research and training notebooks (including training.ipynb).

## Running the Project

See **ReadMe.txt** for step-by-step setup and run commands.

## Training the Model

The CNN model was trained and fine-tuned using the **STL-10 dataset**. The training logic is encapsulated in a Jupyter notebook:

1. Open **notebooks/training.ipynb** (e.g., in Kaggle or Jupyter) and run all cells.
2. The notebook handles data augmentation, L*a*b* color space conversion, and U-Net training.
3. Final weights are saved into the **models/** directory for use by the application.
