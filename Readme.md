# Technical test

Our technical test contain two parts:
- Show your skills part
- Question answering part

## Show your skiils

You will find **train** and **test** set inside folder `dataset`, you will need to make a data analysis and build a sophisticated (on your point of view) classification model (ML or DL). The field that you will need to predict is `ad_customerSpecific_moderationDecision`, of course **test** set doesn't contain this field and you will need to provide us predictions for this dataset at the end.

You can use any fraemwork that you like, there's no restriction. At the end, we are expecting a data analysis (could be just a simple notebook), the code (notebook is also fine) and the predictions for the test set (index column with a column named prediction).

## Question answering

Here is a list of the quesitons, be free to answer to it in order that you prefer.

- How do you manage the training of a model?

The given dataset is highly imbalanced, we have approximatively 94% of the samples in the class 'APPROVED', and only 6% in the class 'REJECTED'.
So to train a model, I should take into consideration the fact that my dataset is highly imbalanced. 
There are some techniques that helps dealing with imbalanced data such as : 
`Undersampling` positive class ( Approved ) or `Oversampling` negative class ( Rejected ).
In my assignement, I avoided modifying training data, but to get the similar effect as `Oversampling` the negative class ( Rejected ), I set the parameter **scale_pos_weights** in XGBoost algorithm to control the balance between positive and negative classes. I set it to **nbr of negative samples / nbr of positive samples**/
Using this parameter, the model learns to classify the rare event better by penalizing any incorrect prediction. 


- How would you split your dataset when you have a supervised learning problem?

I split my traning dataset into two sets ( TRAIN , VAL ). The TRAIN set to train my model, and VAL set to validate it.
I used **Stratified Split** to split the classes proportionally between training and validation sets. 

- How would you evaluate a model (classification)?

Because I have a highly imbalanced dataset. It's normal to have a high Accuracy for the class `Approved`. So I would not rely on the accuracy of that class to validate my model. 
To validate my model, I plot the **ROC Curve** because it helps to see the performance of the model. It plots True Positive Rate against True Negative Rate. 
I also plot the **confusion matrix** to see the Precision/Recall.
With the ROC Curve and the confusion matrix, I can have a better idea how my model correctly classify the samples in both classes.
**F1_score** is also a good metric to evaluate the model because it's a `harmonic mean`of precision and recall.
In the context of detecting `Rejected` ads, I believe that it makes more sence to have a manual reviewer find that the ad is `Approved`, but it is much harder to identify a `Rejected` ad that was never even flagged as such. That why, I'm concerned more with having a good **recall** than **precision**.
For example, for my model: The recall of the positive class `rejected`is high, but the precision is still low. 
That means the class is well detected but the model also include points of other class in it.

