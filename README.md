
=====

# TEGNet: Thermoelectric Generator Modeling and Simulation

This repository provides Python-based tools for training and using the **TEGNet** neural network model to predict the performance of thermoelectric generators (TEGs). Once models for different materials are obtained, users can flexibly combine them to design segmented, n-paired, and even more complex TEG architectures with accelerated speed and high accuracy.

---

## 1. System Requirements

### Software
- Python 3.10
- PyTorch 2.5
- NumPy >= 1.26
- Pandas >= 2.2
- SciPy >= 1.15
- scikit-learn >= 1.6

### Tested Versions
- Python 3.10
- PyTorch 2.5
- Windows 11

### Hardware
- Typical desktop or laptop CPU sufficient
- RAM: ≥ 8GB recommended

---

## 2. Installation Guide

### Using Conda
1. Clone the repository:
```bash
git clone <repository_url>
cd <repository_folder>

2. Create the Conda environment from the provided YAML file:
```bash
conda env create -f environment.yml
conda activate tegnet

3. Ensure the data/ folder contains the required CSV files for training (e.g., Ag2TeS.csv, Bi2Te3.csv) and the property/ folder contains material property CSV files.

Typical install time: 20 minutes on a standard desktop.

## 3. Demo
###Running the Demo
1. Train a model (optional if using pre-trained models):
```bash
python Bi2Te3_model training.py

A trained model will be saved in the model/ folder (e.g., Bi2Te3.pth).

2. Run simulation using pre-trained or newly trained models:
```bash
python single_MgAgSb.py

or for segmented TEGs:

```bash
python segmented_MgAgSb_Bi2Te3.py

Note: The scripts can be adjusted to use different material models, geometries, and boundary conditions.

###Expected Output
CSV files exported to the working directory:

I_scan_Bi2Te3_Th_573.csv	Current sweep data at fixed hot-side temperature
T_scan_Bi2Te3.csv	Maximum power and efficiency vs. hot-side temperature
Length_scan_10mm_MgAgSb_Bi2Te3.csv	Maximum power and efficiency vs. different length ratios of different materials
......

## 4. Instructions for self-adaption
###Current Sweep
```python
I_range = compute_ih_range(
    a=3.5, b=3.5, c=7, Tc=293.15, Th=573.15,
    sigma_fn=sigma_fn, Seebeck_fn=Seebeck_fn
)

V_list, q_list, P_list, efficiency_list, I_list, P_max, eff_max = scan_current_sweep(
    model, a=3.5, b=3.5, c=7, Tc=293.15, Th=573.15,
    I_range=I_range, stats=stats
)

export_current_scan_csv(
    'I_scan_output.csv', I_list, V_list, q_list, P_list, efficiency_list
)

###Temperature Scan
```python
Th_list = np.linspace(313.15, 573.15, 14)

Th_valid, P_max_list, eff_max_list = temperature_param_scan(
    model, Tc=293.15, Th_list=Th_list,
    a=3.8, b=3.8, c=7, sigma_fn=sigma_fn, Seebeck_fn=Seebeck_fn, stats=stats
)

export_temperature_scan_csv(
    'T_scan_output.csv', Th_valid, P_max_list, eff_max_list
)

## 5. Reproducing Results
To reproduce the results reported in the associated manuscript:

1.Prepare the same CSV datasets in the data/ and property/ folders.
2.Train the models using train_model.py.
3.Run the current sweep, temperature scan, or segmented TEG simulation scripts.
4.Exported CSV files contain all quantitative results for plotting or further analysis.


