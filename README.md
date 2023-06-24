# GraphSemiSupLabelProp

<p align="justify">
  This repository contains the implementation of <b>GraphSemiSupLabelProp</b>, a semi-supervised machine learning project developed for the Theory of Graphs course. The project focuses on utilizing label propagation, a powerful algorithm for semi-supervised learning, specifically the part related to label propagation, within the context of graph theory.
</p>

## Project Overview

<p align="justify">
  The goal of this project is to demonstrate the application of label propagation using the <b>make_moons</b> and <b><b>make_circles</b> datasets from the <b>sklearn</b> library in Python, as well as custom CSV datasets obtained from the website <b>statsim</b>. The implemented code propagates labels iteratively based on distance calculations, radial basis function (RBF) kernels, and scaling operations.
</p>

## Features

<p align="justify">
  <ul>
    <li>Implementation of label propagation for semi-supervised learning in the context of graph theory.
    <li>Utilizes <b>make_moons</b> and <b>make_circles</b> datasets from <b>sklearn</b> library.
    <li>Utilizes custom CSV datasets (<b>make_moons</b> and <b>make_spirals</b>) obtained from <b>statsim</b> website.
    <li>Propagates labels iteratively using distance matrix, RBF kernel, and scaling techniques.
    <li>Supports the exploration and experimentation of label propagation algorithms.
  </ul>
</p>
  
## Usage

<p align="justify">To run the code and reproduce the experiments, follow these steps:
  <ol>
    <li>Install the required dependencies: <b>numpy</b>, <b>scipy</b>, and <b>scikit-learn</b>.
    <li>Clone this repository: <b>git clone https://github.com/amorimluiz/GraphSemiSupLabelProp.git</b>.
    <li>Navigate to the project directory: <b>cd GraphSemiSupLabelProp</b>.
    <li>Run the main script: <b>python main.py</b>.
    <li>The results will be stored in corresponding CSV files.
    <li>Feel free to explore the code, experiment with different datasets, and modify the algorithm parameters to gain a deeper understanding of label propagation for semi-supervised learning.
  </ol>
</p>

## Contributing

<p align="justify">Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.</p>

## License

<p align="justify">This project is licensed under the <b>MIT License.</b></p>
