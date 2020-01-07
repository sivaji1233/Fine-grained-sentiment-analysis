# Fine-grained-sentiment-analysis

 # Recurrent Neural Network

## 1.	Representation:
We are using word embeddings to convert every word into a vector of d dimensions. For this we are using the Glove Word embedding which we downloaded from Stanford Glove Site . We used the Glove corpus for both 50d and 300d. We chose Glove over manually initializing the vectors as these vectors are pre-trained on vast corpus and Google News available on the web. We intuitively thought that these vectors would perform better than manually initializing them. In conclusion, we are using pre-trained Glove word embeddings and we are not updating the weights of the word embeddings during the course of the model in the interest of time. 
For our model purpose we are loading the glove.6B.300d.txt file and converting it into a dictionary with unique words as keys and embedding vector as values. We are doing this using our user defined glove_dict_generation() function. 


## 2.	Initialization
Here we are defining weights in two folds.  First, we are using the pretrained weights of Glove embedding and we are not updating these weights in the interest of time. For the RNN we are using the nn.rnn() in pytorch to define our rnn() architecture. Hence, the weights for rnn are defined internally by the model. Also, for the final FFNN layer, we are using nn.linear() which initializes the weights and bias internally. 
We are initializing the hidden dimension vector hx for the RNN as a random vector with values between -0.01 to 0.01. 
 
For each document/review we are creating an embedding matrix which we are passing through the model. This matrix is in the form of a tensor. For creating this matrix we are scanning through the entire Glove dictionary and picking up the vectors corresponding to every word present. For words which are not present, we are randomly assigning a vector for those.  
 
## 3.	Training:
We are splitting the training and validation sets into minibatches. We are playing around with the mini-batch size in the later section.  Inside minibatch we are passing each document in the form of a embedding matrix into the model and getting the predicted vector. We are then calculating the loss w.r.t. The gold label. We are accumulating the loss over the minibatch length and then calculating the gradients using backward() and updating the model parameters using optimization.step(). So basically we are doing backpropagation and updating model weights every minibatch. We find this is a good middle ground between updating model weights after every example and updating model weights at the end of the entire training set.  We later play around with the minibatch size to see how it affects the results. 

For validation, we are not updating any model weights but we are calculating the validation loss to determine our early stopping criterion which we will discuss later. 
 
## 4.	Model:
We used nn.rnn() for the RNN implementation. For the top FFNN later we are using nn.linear(). We are passing the embedding matrix for each document/review into the model in the form of tensor and initializing a hidden vector at the start of the model run. This vector is passed on from the first word to the last word through various FFNNs within the RNN implementation. This part is taken care of internally by pytorch.  We are getting the predicted vector which is of dimension 5 corresponding to our five output review labels. 

## 5.	Linear Classifier
The output of RNN is the vector with dimension h1 corresponds to the last word of the document. This is self.hx in our case as seen in the snippet above. We pass it through the FFNN architecture. 
The FFNN has one hidden layer. The final out is z2 which is passed through the softmax and activation to get a vector of 5 dimensions with probability values. Once we are getting the predicted_vector, we are using the argmax function to pick the position corresponding to the maximum probability to compute the loss for that document. 

## 6.	Stopping:
We used early stopping in our model. We calculate the loss for each epoch in the validation and keep a track of the loss for each epoch. We set a relaxation criteria of 5 which means if for 5 consecutive iterations, the loss doesn’t decrease progressively, we choose to break out of the loop and stop the computation. The following code snippet explains it. We chose validation loss to early stop as we realized that this is the best parameter to track overfitting. If the model is overfitting then the validation loss will not decrease.
 
## 7.	Hyperparameter:
The model as various hyper parameters which are as follows:
h1: This the hidden layer dimension that the rnn is going to output as a vector. We are keeping this as 50 for our base model.
input_dim: This is the input layer dimension to the rnn. This corresponds to the embedding vector   dimension which we imported from Glove* . We are using a dimension of 300.
hx: This is the vector we initialize for the rnn, which is fed from one word into the next in the recurrent format and gets sequentially updated. This has dimension h1. 
h: This is the hidden layer dimension for the FFNN before it is sent to the outer layer. We kept this value as 32 similar to the first part.  
minibatch: We are using mini-batch to run our model in segments. We kept this at 16 in our base model similar to part 1 and played around with this parameter in the model comparison phase. 
number_of_epochs: This is a parameter which we played with in the next part of the report.    
	max_tol_no_improv_epoch: We kept this at 5 as we found it reasonable to give the model 5 
	Chances to improve on the validation loss before terminating.  

# Analysis

Part-3.1 Across-Model Comparison
For comparison of 3 pairs of configuration across models, we have considered following 
1.	Number of Hidden Dimensions (S.No 1,2,3)
2.	Type of Activation function (S.No 1,4,5)
3.	Type of Optimizer (S.No 1,6)

S.No	Hidden Dimensions	Activation	Optimizer	FFNN (Accuracy)	RNN
(Accuracy)
1	32	ReLU	SGD	0.54625	0.32125
2	50	ReLU	SGD	0.540625	0.315
3	20	ReLU	SGD	0.548125	0.295
4	32	tanh	SGD	0.5025	0.24375
5	32	Sigmoid	SGD	0.55	NA 
6	32	ReLU	Adam	0.56875	0.429375

We argue that comparing the two models using the above configurations is a fair comparison since when we are considering a single configuration for example “Activation function”, we are keeping the other configurations same for both the models. 

## Nuanced Quantitative Analysis:

1.	Number of Hidden Dimensions: FFNN performed better when the hidden dimensions is small(20) compared to when it is large(50). Whereas, RNN performed better when the hidden dimension is large(50) compared to when it is small(20). We have tried changing the number of hidden dimensions to 20 and 50 for both FFNN and RNN configurations and reported the results in the above table. For both the cases, there is no clear evidence of model’s performance based on the increase of hidden layers. Normally the model should perform better for lower dimensions since we have less data, in bigger architecture this data will underfit. 

	For S.No 3 of the above table:
		Number of reviews with that FFNN predicted correctly = 790
		Number of reviews what RNN predicted correctly = 318
		Number of intersection reviews from above = 156
		Therefore the FFNN is getting 49% times when RNN is getting correct.

2.	Type of Activation: Both FFNN and RNN performed better when the activation function is Relu  compared to when it is tanh). One of reasons for this is that ReLU non vanishing gradient, which accelerates the convergence of the SGD compared to others. And also we have considered 0 vectors for unseen data, this may create sparse data, ReLU works good in this case too. However we are also using mini-batch SGD, the cons of ReLU which is zero-centered will not affect the accuracy of the model.

For S.No 1 of the above table:
		Number of reviews with that FFNN predicted correctly = 437
		Number of reviews what RNN predicted correctly = 314
		Number of intersection reviews from above = 87
		Therefore the FFNN is getting 28% times when RNN is getting correct.

3.	Type of Optimizer: Both FFNN and RNN performed better when the optimizer is Adam compared to when it is SGD. Since Adam optimizer combines the advantages of two SGD extensions — Root Mean Square Propagation (RMSProp) and Adaptive Gradient Algorithm (AdaGrad) — and computes individual adaptive learning rates for different parameters, which gives us an intuition that it would perform better which is further evident from the results. 
	
For S.No 6 of the above table:
		Number of reviews with that FFNN predicted correctly = 683
		Number of reviews what RNN predicted correctly = 413
		Number of intersection reviews from above = 170
		Therefore the FFNN is getting 41% times when RNN is getting correct.

## Within Model Comparison

For the within model comparison, we considered 4 models:
1)	Baseline model
2)	Baseline model with modification A (Here we changed the mini bath size)
3)	Baseline model with modification B (Here we changed the learning rate)
4)	Baseline model with both modifications A and B(Here we changed both mini batch size and learning rate)

The results for all the 4 models are reported in the table below:

Mini Batch Size	Learning Rate	RNN
(Accuracy)
5	0.05	0.199375
16	0.05	0.226875
5	0.01	0.286875
16	0.01	0.32125

We observed that there was an increase in the RNN accuracy when we made our first modification by increasing the mini batch size from 5 to 16. We also  observed that RNN accuracy increased when we reduced the learning rate from 0.05 to 0.01. Also, combining both these modifications by increasing the mini batch size(from 5 to 16) and reducing the learning rate(from 0.05 to 0.01), we achieved the best accuracy out of all the 4 models.

## Nuanced Qualitative Analysis:

For nuanced qualitative analysis, we collected individual examples which displayed characteristics as below:

1)	An example that all the models predicted correctly
My family has been looking forward to giving this new BBQ restaurant here in Surprise a try for a long time.  After countless delays, and missed opportunities due to the restaurant being closed we had our chance!...........................I rate "Got Que? " a two out of five based on the lean quality of the pork and friendly staff.  Overall however I just literally throw $25 in the trash, and just don\'t see a reason to return.  Sadly, if you want quality "Que", you will have to keep making the drive to outside of Surprise.' 
True label : 1 

This article was predicted correctly by all the configurations of RNN. We suspect that this article has a lot of very popular words with negative connotations which makes it relatively easier for an RNN to predict it correctly with less tuned parameters, in this case a mini-batch size of 5 and a learning rate of 0.05. The words we suspect contributed the most towards its correct prediction are as follows: closed, limited, ‘throwing everything out’ (as a context), stale, ‘no flavor’ etc.
We also suspect that the final few words  of the review have a strong negative connection making it easier for the RNN to predict the rating of the review. 

2)	An example that all the models predicted incorrectly
'I stopped here on a stormy Tuesday thinking it might be less busy and I could easily find a parking space…...In order to survive here they need to be open to people over 40, improve their bar service and make their restaurant a destination where people will walk for  blocks to get there”
True Label : 1

This article was given 1 start but it was written subtly with a lot of description of the situation etc. Hence, it was hard for the RNN even with all the hyper parameter tuning was done on the model. Also this article is very long. Since we are sending the entire article through the RNN at once, we feel that the gradients almost vanish during backpropagation. Long term dependencies are lost in this case leading to misclassification. In this article, we feel that due to the sentence structure, long term dependencies are essential to weed out the negative meaning in a holistic sense.

3)	An example that Model 4 predicted correctly, but others didn’t
"Service here is hit and miss.  We sat in the far section, waited for 20 minutes and then finally had to find a server for assistance…...I would expect better. \n\nThat was our second time, our first experience there was much better. The only reason I gave more than 1 star."
True Label : 1

This article was predicted correctly by the best configuration of RNN(with learning rate=0.01 and mini-batch size of 16). With a minimatch size that is not too small and not too big, we are updating our weights multiple times per epoch. Hence we are ideally moving towards better accuracy. With a learning rate of 0.01, we are learning in small increments which means we are capturing subtle indications of positive/negative sense. Also, the article has quite blunt negative connotative words in shorter spans making it easier for the network to learn. The last line of this review is very affirmative of negative sense hence which helps the RNN to classify it accurately as the vanishing gradients issue still doesn’t exist. 

Part 4:  Questions

Q1 RNNs are known to struggle with long-distance dependencies. What is a fundamental reason for why this is the case?

Answer: RNNs are known to struggle with the long range dependencies mainly due to the vanishing gradient problem and the linear decay of the context information.  Due to the chain rule, the magnitude of gradients are affected by the weights and the derivatives of the activation functions, which are recurrently appearing in the network. The gradients may vanish in time, if either of these factors is smaller than 1 since we multiply small values together and the gradients might increase a lot if these factors are larger than 1. In both cases we lose information especially when the text is longer in length. If there are small gradients during training, it translates to insignificant changes in the weights and thus, there is no significant learning.

Q2) Feeding a sentence into an RNN backwards (i.e. inputting the sequence of vectors corresponding to (course, great, a, is, NLP) instead of (NLP, is, a, great, course)) tends to improve performance. Why might this be the case?

Answer:  Reverse RNN can improve performance in cases of next word prediction and sequence tagging problems. In Language modelling using RNN , the dependency range decreases with diminishing gradient. Hence, if we want to predict the next word in a sentence which has a greater dependency on the first word rather than the word before it, then a forward feed will perform poorly as compared to reverse feel. Let’s take an example : 
“Erik is a good man”. In this case if we want to predict man then the word Erik has greater dependency compared to the word good. Hence, in this case if we train the model with good a is Erik as compared to Erik is a good, we have a greater chance of predicting man in the former case (reverse feed case).
Similar is the case with sequence tagging. Example POS tagging. Consider the sentence:
“The old man the ship”. If we want to predict the POS of man by forward pass, we will assign it a noun compared to if we use a reverse pass, then the RNN will assign it as a verb. However, when we see the entire sequence, we see that verb is the right tag. Therefore, reverse RNN is better in this case.

Q3) In using RNNs and word embeddings for NLP tasks, we are no longer required
to engineer specific features that are useful for the task; the model discovers
them automatically. Beyond concerns of dataset size (and the computational resources required to process and train using this data as well as the further environmental harm that results from this process), why might we disfavor RNN models?

Answer: Apart from the main disadvantage of RNN which is the gradient vanishing and exploding problems which can be solved by LSTM, the other problem is that RNN cannot be stacked into very deep models. This is mostly because of the saturated activation function used in RNN models which makes the gradient decay over layers.

Moreover, Deep Learning models such as RNNs work well when there is abundance of training data such as hotel review classification, movie review predictions etc. However, when the training is done on small datasets very specific to specialized tasks (eg. decoding sentences etc) RNNs will not perform well as we will not have enough training data to run it on. 



# References:
https://medium.com/explore-artificial-intelligence/an-introduction-to-recurrent-neural-networks-72c97bf0912 [Part 2]
https://medium.com/@xtaraim/a-very-basic-introduction-to-feed-forward-neural-networks-97a33b34604f [Part 2]
https://pytorch.org/docs/stable/nn.html



