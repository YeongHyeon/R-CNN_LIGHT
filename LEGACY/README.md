<h1>R-CNN_LIGHT</h1>

<img src="sample.gif">
<p>Example of eye blink recognition.</p>
<p><strong>Maximum speed: 450 classification/sec</strong></p>

<h2>Requirments</h2>
<p>Python 2.7</p>
<p><a href="http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_setup/py_setup_in_windows/py_setup_in_windows.html">OpenCV</a></p>
<p><a href="https://www.tensorflow.org/install/">TensorFlow</a></p>
<p>Numpy, Matplotlib ....</p>

<h2>common\</h2>
<p>This directory contains several functions.</p>
<p>The original is located <a href="https://github.com/oreilly-japan/deep-learning-from-scratch">here.</a> and I have added a couple of functions that I need or a little modification as needed.</p>

<h2>dataset\</h2>
<p>This directory will organize automatically.</p>

<h2>images\</h2>
<p>Put your own Image datas in this directory.</p>
<p>0: class0, 1: class1 ....</p>

<h2>params\</h2>
<p>This is the directory where the output of the train is stored.</p>
<p>It is possible to reproduce a plurality of images in one training, without training each time the image is reproduced.</p>

<h2>custom_convnet.py</h2>
<p>It contains Convolution Neural Network.</p>
<p>The structure is as follows.</p>
<p>Layer1: Conv-ReLU-Pool</p>
<p>Layer2: Conv-ReLU-Pool</p>
<p>Layer3: Conv-ReLU-Pool</p>
<p>Layer4: Affine(Fully connected)</p>
<p>The weights are initialized by 'He initialization'.</p>

<h2>dataset_loader.py</h2>
<p>It requires <a href="https://www.tensorflow.org/install/">TensorFlow</a></p>
<p>Because useing the function 'create_image_lists'</p>
<p>The original 'create_image_lists' function is <a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py">here.</a></p>

<h2>run_r-cnn.py</h2>
<p>Enter the commend on your terminal like this.</p>
<p><strong>$ python run_r-cnn.py</strong></p>
<p>If you want to know about options ... </br><strong>$ python run_r-cnn.py -h</strong></p>
<p>To use video as source instead of webcam ... </br><strong>$ python run_r-cnn.py --source {video source path}</strong></p>
<p>If you want to train for the new data ... </br><strong>$ python run_r-cnn.py --train_epochs {How many times to repeat as a number} --batch_size {Batch processing size as a number}</strong></p>

<p>
Last Update: 2017.06.24
</p>
