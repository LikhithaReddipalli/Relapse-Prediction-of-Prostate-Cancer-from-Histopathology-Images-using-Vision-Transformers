import torch
import torch.nn.functional as F


def test_bce_with_logits_loss():

    # Define the ground truth labels for the test case
    labels = torch.tensor([1, 0, 1, 0])
    
    # Define the predicted scores for the test case
    scores = torch.tensor([3.0, 1.0, 2.0, -1.0])

    # Compute the expected loss using the binary cross-entropy loss formula
    expected_loss = (-torch.log(torch.sigmoid(scores[0])) 
                     -torch.log(1 - torch.sigmoid(scores[1]))
                     -torch.log(torch.sigmoid(scores[2]))
                     -torch.log(1 - torch.sigmoid(scores[3]))) / 4.0

    # Compute the actual loss using the BCEWithLogitsLoss function
    loss_fn = torch.nn.BCEWithLogitsLoss()
    actual_loss = loss_fn(scores, labels.float())

    # Compare the expected and actual loss values
    assert torch.allclose(actual_loss, expected_loss, rtol=1e-3)


def test_bce_with_logits_loss_weighted():

    # Define the ground truth labels for the test case
    labels = torch.tensor([0., 0., 1., 0.])
    
    # Define the predicted scores for the test case
    scores = torch.tensor([ 0.3946, -0.0776,  0.0514,  0.1777])

    # Define the weights for the positive and negative examples
    pos_weights = torch.tensor(2)
    neg_weights = torch.tensor(1)

    #pos_weight = torch.tensor(100)
    #neg_weight = torch.tensor(1)

    weights_tensor = labels * pos_weights + (1 - labels) * neg_weights

    #class_weight = torch.tensor([neg_weight, pos_weight])

    # Compute the expected loss using the weighted binary cross-entropy loss formula
    expected_loss = ((-torch.log(1- F.sigmoid(scores[0])) * (1-labels[0]) * neg_weights
                      -torch.log(1 - F.sigmoid(scores[1])) * (1 - labels[1]) * neg_weights
                      -torch.log(F.sigmoid(scores[2])) * labels[2] * pos_weights
                      -torch.log(1 - F.sigmoid(scores[3])) * (1 - labels[3]) * neg_weights) / 4.0)

    # Compute the actual loss using the BCEWithLogitsLoss function with weights
    loss_fn= torch.nn.BCEWithLogitsLoss(pos_weight= pos_weights)
    actual_loss = loss_fn(scores, labels)

    print('expected loss_1', expected_loss)
    print('actual loss_1', actual_loss)

    # Compare the expected and actual loss values
    assert torch.allclose(actual_loss, expected_loss, rtol=1e-3)



#test_bce_with_logits_loss()
test_bce_with_logits_loss_weighted()