# GDA - Geometric Data Analysis Project

## Auteurs : Guillaume Lévy, Lucas Versini

Ce dépôt contient le code pour notre projet réalisé dans le cadre du cours de [Geometric Data Analysis](https://www.jeanfeydy.com/Teaching/index.html) enseigné par Jean Feydy.

## Prérequis
Les bibliothèques nécessaires pour exécuter ce projet sont :
- `torch`
- `numpy`
- `matplotlib`

## Instructions
Pour évaluer des templates en se basant sur les images du dossier `data`, exécutez simplement `python main.py`

## Description des fichiers
- `main.py` : s'occupe de l'estimation des templates.
- `trainer.py` : contient le code principal pour mettre à jour les différents paramètres du modèle (algorithme espérance-maximisation).
- `optimize.py` : regroupe les outils nécessaires à l'optimisation d'un des paramètres par minimisation d'une fonction objectif.

## Organisation des données
Les images utilisées doivent être placées dans le dossier `data`.
```
data/
│
├── template_0/
│   ├── train/
│   │   ├── 1.png
│   │   ├── 2.png
│   │   └── ...
│
├── template_1/
│   ├── train/
│   │   ├── 1.png
│   │   ├── 2.png
│   │   └── ...
│
├── template_.../
│   ├── train/
│   │   ├── 1.png
│   │   ├── 2.png
│   │   └── ...
```
