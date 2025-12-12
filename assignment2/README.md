# Assignment 2 — Adversarial Attacks vs. Hill-Climbing Search

**DSAIT4 – Testing AI Systems**

This assignment extends your work on hill-climbing–based adversarial image generation by introducing standard adversarial attack baselines using the CleverHans library (FGM + PGD).
You will compare your method against these attacks on the same set of images.
    

## Repository Structure
```
├── baselines.py                # Provided: CleverHans FGM + PGD attacks on student images
├── hill_climbing.py            # YOU implement and complete this file
├── images/                     # Input images to attack
│    ├── fish.jpg
│    ├── castle.jpg
│    ├── ...
├── data/
│    ├── image_labels.json      # Human-provided labels for the images
│    ├── imagenet_classes.txt   # List of ImageNet classes
├── requirements.txt
└── README.md
```

## Installation

This code has been tested with Python 3.9.21 and on MacOS (26.1) and Linux machine. 
The use of GPU is optional as the code can work on CPU. However, the model inference time will be slower on CPU.

We strongly recommend using a virtual environment (e.g., ``venv``):

```bash
python3 -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows PowerShell
```

To install dependencies, run the command:

```bash
pip install -r requirements.txt
```

## Running the baselines:

The baselines are implemented by CleverHans library  and can be executed via```baselines.py```.  
The script will:

* Load each image listed in data/image_labels.json
* Predict the clean label using VGG-16
* Apply FGM and PGD attacks from CleverHans
* Save attacked images to attack_results/
* Print prediction shifts for clean vs adversarial samples

You can simply run the baselines with:

```bash
python baselines.py
```

And it will produce output like:

```Image: fish.jpg 
Human label: goldfish                       --> This the seed image for the attack
Model prediction (clean): goldfish (0.949)  --> Original prediction (with confidence level)
FGM prediction: stole (0.255)               --> Prediction of the attack image by FGM (with confidence level)
PGD prediction: Maltese dog (1.000)         --> Prediction of the attack image by PGD (with confidence level)
````


## Student Task — Implement the Hill-Climbing Search

The file ``hill_climbing.py`` contains an incomplete implementation. Students must complete the implementation by:

* Design mutation operators for altering the pixels values 
* Implement a fitness function measuring how close the target model (VGG16) is to make a wrong prediction
* Implement the hill-climbing loop
	•	Generate perturbed images
	•	Compare against the baselines

Note that the hill climber must implement a **black-box** strategies to generate adversarial: the fitness function (which is 
guidance for the attack) should only consider the input (mutated images) and output (predicted labels with confidence 
value). 

The HC must read input exactly like the baseline:

* For each image in ``data/image_labels.json``
* Load images from ``images/``
* Use the human label (``label`` attribute for each entry in ``image_labels.json) for correctness evaluation


**Critical**: A successful attack is defined as any perturbation to the input image that remains within the 
ε-bounded L∞ constraint (no pixel changes by more than ε) and causes the VGG16 model’s top-1 predicted class to change.


