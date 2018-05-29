# tensorglove
A hand pose recognition library using Tensorflow, for use with the Noitom Hi5 gloves. 
This project was the result of 2 days playing around with the gloves and tensorflow, and so is extremely preliminary. 

## Getting Started

The repository consists of two components: a Unity project for performing the hand tracking in VR, and a set of python scripts for performing
the machine learning. 

To train the neural network and run the server, the following packages are required:

* Tensorflow 1.7
* python-osc (for running the server). 
* pandas 

Simply run the script `tensorglove.py` to train the neural network. After training, by default it will run an OSC server to provide predictions
to the Unity project. 

To run the Unity project, the following plugins will need to be added to the project (in the glovetest/Assets folder): 

* [Noitom Unity package](https://hi5vrglove.com/downloads)
* [SteamVR Unity plugin](https://assetstore.unity.com/packages/templates/systems/steamvr-plugin-32647) 

Open the Unity project (root at the folder `glovetest`) and open the GloveGestureTest scene. 

The `Hi5CsvRecorder` script in the hierarchy can be used to record poses for use in training the neural network. Select which gesture you 
are training for (currently from None, Click, Fist and Point) and press L or R to record data for the left or right hand respectively. 

The `TensorflowOscGesture` script can be used to communicate with the tensorglove server. The basic demo only works for the right hand and 
changes the colour of the hand based on the gesture detected.  

## Contributing 

The following is a list of ideas for this project.

* More training data. 
* Train the left hand! 
* Optimize the neural network, tuning hyperparameters. 
* Make the gesture control do something!
* Figure out how to serve the trained [tensorflow model](https://www.tensorflow.org/serving/serving_basic) correctly. 
* Explore other/simpler machine learning solutions with [scikit-learn](http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html), or [C# SVM](https://github.com/ccerhan/LibSVMsharp).
* Explore Microsoft's C# [neural network offering](https://docs.microsoft.com/en-us/cognitive-toolkit/using-cntk-with-csharp). 
