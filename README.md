# Suffix Prediction in Process Mining

## An Introduction to Deep Learning-based Suffix Prediction in Predictive Process Monitoring 

---

**Author**: Pedro Gamallo Fernandez --- *University of Santiago de Compostela*

**Email**: pedro.gamallo.fernandez@usc.es

**Last review**: July 2024

(In development)

---

## Summary

This repository collects the main state-of-the-art deep learning based approaches for 
suffix event prediction in the field of process mining. The approaches of each author are 
grouped in explanatory notebooks that allow a step-by-step execution, including data 
preprocessing, model training and prediction process, in order to better 
understand the proposed solution.

The authors included are:

- N. Tax [1].


## Installation and usage
For local use it is necessary to clone or download the repository from 
https://gitlab.citius.usc.es/pedro.gamallo/suffixoprediction 
and configure a virtual environment (*Conda* recommended) for the installation of the 
necessary packages:

- Create the virtual environment with a recent Python version (3.10 or 3.11):
```bash
(base) $> conda create -n suffixprediction python==3.11 
(base) $> conda activate suffixprediction
```

- Install the jupyter package to execute the notebooks:
```bash
(suffixprediction) $> pip install jupyter
```

- Install *verona*:
```bash
(suffixprediction) $> pip install verona
```

- Install required frameworks to execute the models:
```bash
(suffixprediction) $> pip install torch torchvision torchaudio
(suffixprediction) $> pip install tensorflow==2.13
```

- To execute the notebooks, you can open the notebooks in an IDE with Jupyter 
support, or run the following command in console and access from the browser to the 
specified url:
```bash
(anomalydetection) $> jupyter notebook
```

## Bibliography

[1] Tax, N., Verenich, I., La Rosa, M., & Dumas, M. (2017). *Predictive business process 
monitoring with LSTM neural networks*. In Advanced Information Systems Engineering: 
29th International Conference, CAiSE 2017, Essen, Germany, June 12-16, 2017, 
Proceedings 29 (pp. 477-492). Springer International Publishing.