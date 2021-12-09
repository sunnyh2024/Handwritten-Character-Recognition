# Handwritten-Character-Recognition

<h3>OVERVIEW</h3>
<p>Welcome to our Handwritten Digit Recognizer! Using the MNIST and EMNIST datasets, we created a model that takes in a 28x28 image and classfies at as an uppercase letter A-Z or a number 0-9. We created this model using the Tensorflow and Keras libraries, and also used a handful of other libraries to read and parse the data (pandas, numpy, etc). The GUI was built using the TKinter library. </p>

<h3>CHALLENGES</h3>
<p>The main challenges that we faced during this process were with the data and the model. We started off just implementing the digits with the prebuilt SGD optimizer in Keras, which worked pretty smoothly. However, once we tried to add the letters as well, we found it was difficult to merge the two csv files, and we had to manually create our own dataset. We trimmed each class so that they were more balanced, and also had to update some labels so that they matched with our model. Another challenge was when we tried to implement our own version of SGD. Although we are pretty sure that the logic behind our SGD class is correct, we were initially unable to feed it into our model due to compatibility issues. A more in-depth explanation can be found in both SGD files.</p>

<h3>HOW TO USE</h3>
<p>This program is interactive, and using it is very simple. Simply run the gui code (either digit_only or combined). Write the character in the prompted box and click recognize to feed your image into the model. To reset the canvas, click clear.</p>

</p>Thanks!</p>
</p>Akshay, Mario, Sunny</p>
