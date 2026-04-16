# Digit Recognition CNN App

This is a fresh MNIST digit recognition project built with:

- TensorFlow/Keras CNN
- Streamlit web app
- image upload prediction flow
- training curves
- confusion matrix
- class-wise report
- misclassified sample visualization
- feature-map visualization
- Render deployment support

## Project Commands

Build command:

```powershell
py -3.11 train.py
```

Start command:

```powershell
py -3.11 -m streamlit run app.py
```

## Render Commands

Build command:

```bash
pip install -r requirements.txt && python train.py
```

Start command:

```bash
python -m streamlit run app.py --server.address 0.0.0.0 --server.port $PORT
```

Environment variable:

```text
PYTHON_VERSION=3.11.11
```
