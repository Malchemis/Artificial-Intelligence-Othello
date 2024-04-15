# AI applied to Othello-Reversi

## How to run the code
1. Clone the repository or get the source code.
2. Download the required libraries using the following command:
```bash
pip install -r requirements.txt
```
IMPORTANT: The python type hints and other features forces the use of python 3.11 or higher.
We recommend creating a virtual environment to avoid conflicts with other projects.

3. Define your own configuration in the `config.py` file.
4. Run the main file using the following command

If you want to redo the statistics obtained in the report, 
you can uncomment the line 132 in the `main.py` file, define 
your own trial run functions or reuse the ones provided commented in the file.

```bash
python path/to/main.py --config path/to/config.yaml
```

To run the notebooks, you will need some more dependencies, install Jupyter Notebook. In particular,
for the model pipeline, you will need to install cuda, pytorch, and have the 
drivers installed.
