### Random notes

PairModel() : 
* When we first created our model, no matter what learning rate we chose it would always underfit to a local minima and systematically yield the same training losse and results on the test set (most likely an issue with gradient vanishing)
* Adding batch normalization fixed the issue and yield an accuracy of ~83%

Siamese():
* Create two parallel models that train on images on both images separately with weights sharing, and have a final layer that combine both layers. Training is done using auxiliary losses on the two parallel models and the loss of the final model.
* Testing : We evaluate two accuracy : the accuracy of the final output with respect to the target class, and the output of the two parallel models where we manually evaluate the target by comparing the two outputs => We get a better accuracy on combining the output of the two parallel models than with the final linear + ReLu() steps (probably not a single linear steps to go from the 20 concatenated probabilities to the final output?)

