# Multiclass_Classification_MaximumEntropy
 we have to build a linear model using XOR.data. Here first
one_hot_encoding is done where the y is predicted based on the number of classes present. So,
here in the xor.dat there are two classes based on that each row gets classified and returns the
one_hot_encoding for the data.
As given in the pdf softmax, loss and gradient functions are calculated. The hyper parameters
used in our code are alpha, epsilon, lambda, epoch(iterations).
In q1a.py, the parameters taken are
W and b are the weights of the model.
Here the weights are taken with random values. In this case, the values taken are
For every iteration, the softmax function is calculated with the updated weights and then the
gradient function is called where the normal W and b gets updated to dw,db by multiplying the
values with alpha and updating them.
The linear classifier cannot solve the xor data so the accuracy is very less.
The loss vs epoch graph comes out to be like this for the above parameters.
The output for which the difference between the updated gradient and gradient rules given in the
pdf comes out to be CORRECT.

Here, a multi-class data is tried to fit into the model that we earlier built. The
hyperparameter here, W and b dimensions are changed. After updating the gradient values for W
and b. Three lines are taken where the W and b are updated.

The spiral_data.dat is plotted and accordingly the decision boundaries are plotted.The second
class and first class decision boundary is almost close enough on the plotted graph. The decision
boundaries are plotted as much as possible to classify the different classes in the data.
The accuracy obtained here is low as the spiral data cannot be linearly classified

mini batches are created with a size of 50. With the mini batches
created, each and every batch in the batches list are taken and the weights bias and new x and y
are taken from the batches. With the mini batches, the loss is plotted and for the normal
initialized values of W and b, the loss_test is plotted.
