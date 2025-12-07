# Atlantico Processing

This project provides a Scikit-learn based pipeline for preprocessing the PAMAP2 dataset (or similar).

## Structure

- `src/loader.py`: Handles loading of `.dat` files using Pandas. Supports chunking for large files.
- `src/transformers.py`: Custom Scikit-learn transformers for filtering and interpolation.
- `src/pipeline.py`: Defines the preprocessing pipeline.
- `preprocess.py`: Main script to demonstrate loading and processing.

## Usage

1.  Ensure your dataset is in `dataset/Protocol/`.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the preprocessing script:
    ```bash
    python preprocess.py
    ```

## Customization

You can modify `src/pipeline.py` to add more steps or change the existing ones.
You can select different columns in `preprocess.py`.
