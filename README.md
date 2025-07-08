# OpenHands_double_pendulum

A Python simulation of a double pendulum, based on the specification in `Specification.md`.

## Dependencies

- numpy
- matplotlib
- scipy

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the simulation and visualization with:

```bash
python double_pendulum.py
```

You can enable saving outputs by setting the flags at the top of `double_pendulum.py`:

```python
# At the top of the file
SAVE_GIF = True     # Save animations as GIF
SAVE_VIDEO = True   # Save animations as MP4
SAVE_PLOTS = True   # Save static trajectory plots as PNG
```