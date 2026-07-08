# ML-Driven-Battery-Prognostics-and-Eco-Impact-Assessment-using-ADALM1000
<p align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange?logo=scikitlearn)
![GUI](https://img.shields.io/badge/GUI-Tkinter-green)
![Platform](https://img.shields.io/badge/Platform-Windows-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![Hardware](https://img.shields.io/badge/Hardware-ADALM1000-red)
![Battery](https://img.shields.io/badge/Battery-18650%20Li--Ion-success)
![Maintained](https://img.shields.io/badge/Maintained-Yes-success)
![Contributions](https://img.shields.io/badge/Contributions-Welcome-blueviolet)

</p>
This project is a complete machine-learning and GUI-based battery health analysis system for 18650 lithium-ion cells. It combines a trained predictive model, a battery dataset, and optional live hardware measurement from an ADALM1000 instrument to estimate:

- the likely failure mode of a battery,
- its remaining useful life (RUL),
- state of health (SoH),
- sustainability and eco-impact indicators,
- and replacement urgency.

The application is implemented in [MODEL_AND_PREDICTION.py](MODEL_AND_PREDICTION.py), and the optional hardware interface is implemented in [dataAcquisition.py](dataAcquisition.py). Pretrained model artifacts are already provided in [models/](models/).

---

## 1. What this project is

This project solves a practical battery diagnostics problem: instead of relying only on manual inspection or simple threshold rules, it uses machine learning to analyze battery-related features and estimate how a battery is degrading.

In simple terms, the system answers questions like:

- Is this battery likely to fail due to SEI growth, lithium plating, electrode cracking, or normal aging?
- How many cycles might it still last?
- Is the battery healthy enough for continued use?
- What is the environmental burden and replacement risk?

### Core idea

The pipeline uses:

1. a dataset of battery measurements and known failure modes,
2. preprocessing and feature scaling,
3. machine-learning models for classification and regression,
4. a graphical interface for interactive prediction,
5. and optional hardware data collection from an ADALM1000 device.

### What the project contains

- [Battery_18650_Dataset.csv](Battery_18650_Dataset.csv): training dataset with 1,050 rows and 27 columns.
- [MODEL_AND_PREDICTION.py](MODEL_AND_PREDICTION.py): full GUI application for training, prediction, visualization, and hardware-assisted acquisition.
- [dataAcquisition.py](dataAcquisition.py): ADALM1000 interface for acquiring live battery-related parameters.
- [models/](models/): pretrained model files for inference.

---

## 2. Project goals

The project is designed for:

- battery prognosis and health monitoring,
- machine-learning-based degradation analysis,
- sustainability and eco-impact assessment,
- educational and research use with ADALM1000 hardware,
- and a visual desktop application that does not require writing code for normal use.

---

## 3. What the model predicts

The trained system predicts two major things:

### A. Failure mode classification
The classifier predicts one of four battery failure modes:

- SEI_growth
- lithium_plating
- electrode_cracking
- normal_aging

### B. Remaining useful life (RUL) regression
The regressor estimates the remaining useful life in cycles.

### Additional derived outputs
The application also calculates:

- SoH (State of Health)
- Health Index
- Environmental impact score
- Carbon burden estimate
- Replacement urgency recommendation

---

## 4. Repository structure

- [Battery_18650_Dataset.csv](Battery_18650_Dataset.csv) — input dataset used for training and evaluation
- [MODEL_AND_PREDICTION.py](MODEL_AND_PREDICTION.py) — main application
- [dataAcquisition.py](dataAcquisition.py) — optional hardware acquisition layer
- [models/](models/) — trained model artifacts

---

## 5. Software requirements

### Recommended environment

- Python 3.9 or newer
- Windows 10/11 recommended for the GUI workflow
- A working installation of Tkinter (usually included with Python on Windows)

### Required Python packages

Install the following packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib scipy
```

### Optional hardware dependency

If you want to use live ADALM1000 measurements, also install:

```bash
pip install pysmu
```

If you do not install pysmu, the application still works in software-only mode using the built-in dataset and sample/default values.

---

## 6. First-time setup

### Step 1: Clone or download the repository

Open a terminal and go to the project folder:

```bash
cd "F:\Projects\Machine Learning\Project Files"
```

### Step 2: Create a virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib scipy
```

If you plan to use hardware:

```bash
pip install pysmu
```

### Step 4: Run the application

```bash
python MODEL_AND_PREDICTION.py
```

The GUI should open. If the hardware dependency is missing, the app will still run, but the hardware button will be disabled with a warning.

---

## 7. First-time use guide

When you run the app for the first time, you can use it in two ways.

### Option A: Use the existing trained models (easiest)

No training is required. The pretrained models already exist in [models/](models/), so you can:

1. launch the application,
2. go to the prediction tab,
3. enter or load battery values,
4. click predict.

This is the best option for first use.

### Option B: Train the models yourself

If you want to retrain the models on the dataset:

1. open the application,
2. go to the Train tab,
3. specify the dataset path if needed,
4. start training,
5. wait for the model artifacts to be saved in [models/](models/).

The training workflow uses Gradient Boosting for:

- failure mode classification,
- and RUL regression.

---

## 8. How to use the application

The application has multiple tabs:

### Dashboard
Shows dataset statistics and overview information.

### Prediction
Use this to make a prediction for a single battery based on measured or entered parameters.

### Batch
Use this to predict for multiple batteries from a CSV file.

### Visualize
Creates charts and visual analysis of battery conditions and predictions.

### Train
Retrains the models using the included dataset.

### About
Displays project documentation and architecture details.

---

## 9. Hardware workflow with ADALM1000

The project supports optional live acquisition from an ADALM1000 device. The implementation is designed for EIS-style measurement and battery parameter extraction.

### Important note

The current hardware workflow in [dataAcquisition.py](dataAcquisition.py) is intended for measurement and parameter acquisition. It is not a charger/discharger controller. It is used to gather live battery features for prediction.

### When to use hardware mode

Use hardware mode if you want to:

- collect live battery measurements from a real cell,
- fill the prediction form automatically from hardware,
- and compare live measurements against the machine-learning model.

### ADALM1000 connection instructions

The software expects a simple two-terminal connection to the battery under test.

#### Recommended wiring

- Battery positive terminal → ADALM1000 Channel A connection (used as the positive measurement path)
- Battery negative terminal → ADALM1000 Channel B connection (used as the return/reference path)

A simple schematic is:

```text
Battery + ----> CHA+
Battery - ----> CHB
```

#### Practical wiring notes

- Use the red lead for the positive battery terminal.
- Use the black lead for the negative battery terminal.
- Make sure the battery is connected securely and not shorted.
- Double-check polarity before powering or measuring.
- Do not connect damaged or swollen cells.

### What the application does with hardware

When you click the hardware button in the GUI:

1. the program tries to connect to the ADALM1000,
2. performs an EIS-style measurement sweep,
3. extracts battery parameters,
4. populates the prediction fields,
5. and runs the model using those values.

---

## 10. Battery connection and safety notes

This project is intended for educational, experimental, and research use. Please follow these safety precautions:

- Only use a battery that is in safe physical condition.
- Do not use swollen, damaged, or leaking cells.
- Avoid short circuits.
- Use proper leads, clips, and connectors.
- Keep the setup away from flammable materials.
- Do not exceed the battery’s safe operating conditions.
- If you are unsure about the hardware setup, stop and verify the connection before proceeding.

### Recommended battery type

The project is configured around a typical 18650 lithium-ion cell with values similar to the included dataset:

- nominal chemistry: Li-ion / 18650 style cell,
- rated capacity: approximately 2600 mAh,
- full voltage: about 4.20 V,
- empty voltage: about 3.00 V.

The example configuration in [dataAcquisition.py](dataAcquisition.py) uses the following assumptions:

- rated capacity: 2600 mAh,
- full voltage: 4.20 V,
- empty voltage: 3.00 V.

---

## 11. Dataset overview

The included dataset contains 1,050 samples and 27 columns. The main labels include:

- battery_id
- chemistry
- failure_mode
- cycle
- health
- degradation_rate
- RUL
- temperature
- soc
- electrical impedance-related features
- and other battery health descriptors

The classification target is the failure mode, and the regression target is RUL.

---

## 12. How the training pipeline works

When the training process runs, the program:

1. loads the dataset,
2. encodes the failure mode labels,
3. preprocesses the numeric features,
4. splits the data into train/test sets,
5. trains the classifier and regressor,
6. evaluates the model performance,
7. and saves the trained objects in [models/](models/).

The current implementation uses Gradient Boosting models for both tasks.

---

## 13. Expected output

After running the script, you can expect:

- a GUI window to open,
- model training logs or prediction outputs,
- inferred failure mode and RUL values,
- health and sustainability metrics,
- and optionally hardware-acquired measurements.

On a normal local run, the training process reports the classifier accuracy and the regressor RMSE/R² metrics.

---

## 14. Troubleshooting

### Problem: the app does not start

Check the following:

- Python is installed correctly.
- Dependencies were installed successfully.
- Tkinter is available.

### Problem: hardware button is unavailable

This usually means that pysmu is not installed. The software-only workflow still works.

### Problem: no hardware is detected

Check:

- USB cable connection,
- ADALM1000 power state,
- battery wiring polarity,
- and whether the device is visible to the operating system.

### Problem: training fails

Make sure the dataset file [Battery_18650_Dataset.csv](Battery_18650_Dataset.csv) exists and is not corrupted.

---

## 15. Recommended first-run workflow

For a first-time user, the simplest path is:

1. install Python and dependencies,
2. run the application with:

```bash
python MODEL_AND_PREDICTION.py
```

3. use the existing pretrained models in [models/](models/),
4. try a software-only prediction first,
5. then connect the ADALM1000 if you want live measurements,
6. and only then try retraining the models.

---

## 16. Summary

This project is a practical, end-to-end battery prognostics system that demonstrates how machine learning can be applied to battery health monitoring. It combines:

- real battery data,
- predictive models,
- a desktop GUI,
- and optional ADALM1000 hardware measurement.

It is useful for:

- learning machine learning for battery diagnostics,
- testing AI-assisted battery health estimation,
- and exploring eco-impact and replacement planning for lithium-ion batteries.

---

## 17. Quick start command

```bash
cd "F:\Projects\Machine Learning\Project Files"
python MODEL_AND_PREDICTION.py
```
