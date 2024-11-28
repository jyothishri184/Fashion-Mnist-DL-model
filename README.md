# Fashion MNIST image classification with TensorFlow and FastAPI

## Overview

This project demonstrates the development and deployment of a Convolutional Neural Network (CNN) model for classifying images from the Fashion MNIST dataset. The trained model is deployed using FastAPI, allowing users to interactively test the model with custom images. 

## Dataset

The Fashion MNIST dataset consists of 60,000 28x28 grayscale images of 10 fashion categories and a test set of 10,000 images. The categories include T-shirts/tops, trousers, pullovers, dresses, coats, sandals, shirts, sneakers, bags, and ankle boots.



## Quick Start
 
```bash

# Clone the repository
git clone https://github.com/jyothishri184/Fashion-Mnist-DL-model.git

# Navigate to the project directory
cd Fashion-Mnist-DL-model

# Install dependencies
pip install -r requirements.txt

# Run the model
python model.py

# Running the FastAPI Application
uvicorn app:app --reload

```
Making Predictions:

Make predictions by sending POST requests to the /predict endpoint with an image file attached.


## License
This Market Basket Analysis project is licensed under the MIT License.

## Acknowledgments

Special thanks to the open-source community and contributors for inspiration and learning resources.

Happy Coding 🚀
