# Tracking Barbell Exercises with Motion Sensor Data

This project processes motion sensor data from strength exercises using a wearable MetaMotion device. It includes data cleaning, signal filtering, feature engineering, supervised classification, and repetition counting using Python and Jupyter.

---

## Features

- Motion data merging from accelerometer and gyroscope
- Time indexing and resampling (5Hz)
- Signal smoothing using LowPassFilter
- Rep counting via peak detection
- Outlier removal
- Supervised learning (Random Forest classifier)
- Confusion matrix and performance evaluation

---

## Setup Instructions

1. **Clone the repository**  
   ```bash
    git clone https://github.com/mariocroix/S4-MotionRep-Analyzer.git
    cd S4-MotionRep-Analyzer.git

2. **Create conda environment**  
   ```bash
    conda env create -f environment.yml
    conda activate tracking-barbell-exercises

3. **Running the Code (Jupyter Interactive)**  
   All code snippets and scripts can be executed using Jupyter Interactive Mode. This mode was used throughout development:
	-	Open a Python file in VS Code.
	-	Select a code cell and press Shift + Return (Shift + Enter) to execute it.
	-	The output appears in a separate interactive Jupyter panel.
