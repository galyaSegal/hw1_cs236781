import abc
import torch


class ClassifierLoss(abc.ABC):
    """
    Represents a loss function of a classifier.
    """

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    @abc.abstractmethod
    def loss(self, *args, **kw):
        pass

    @abc.abstractmethod
    def grad(self):
        """
        :return: Gradient of the last calculated loss w.r.t. model
            parameters, as a Tensor of shape (D, C).
        """
        pass


class SVMHingeLoss(ClassifierLoss):
    def __init__(self, delta=1.0):
        self.delta = delta
        self.grad_ctx = {}

    def loss(self, x, y, x_scores, y_predicted):
        """
        Calculates the Hinge-loss for a batch of samples.

        :param x: Batch of samples in a Tensor of shape (N, D).
        :param y: Ground-truth labels for these samples: (N,)
        :param x_scores: The predicted class score for each sample: (N, C).
        :param y_predicted: The predicted class label for each sample: (N,).
        :return: The classification loss as a Tensor of shape (1,).
        """

        assert x_scores.shape[0] == y.shape[0]
        assert y.dim() == 1

        # TODO: Implement SVM loss calculation based on the hinge-loss formula.
        #  Notes:
        #  - Use only basic pytorch tensor operations, no external code.
        #  - Full credit will be given only for a fully vectorized
        #    implementation (zero explicit loops).
        #    Hint: Create a matrix M where M[i,j] is the margin-loss
        #    for sample i and class j (i.e. s_j - s_{y_i} + delta).

        # ====== YOUR CODE: ======
        rows = list(range(x_scores.size()[0]))
        marginal_hinge_loss = x_scores - x_scores[rows, y].unsqueeze(1) + self.delta
        marginal_hinge_loss[rows, y] = 0
        marginal_hinge_loss[marginal_hinge_loss < 0] = 0
        loss_per_sample = torch.sum(marginal_hinge_loss, dim=1)
        loss = sum(loss_per_sample) / len(loss_per_sample)
        # ========================

        # TODO: Save what you need for gradient calculation in self.grad_ctx
        # ====== YOUR CODE: ======
        self.grad_ctx = dict({'marginal_hinge_loss': marginal_hinge_loss, 'y': y,
                              'x': x, 'rows': rows})
        # ========================

        return loss

    def grad(self):
        """
        Calculates the gradient of the Hinge-loss w.r.t. parameters.
        :return: The gradient, of shape (D, C).

        """
        # TODO:
        #  Implement SVM loss gradient calculation
        #  Same notes as above. Hint: Use the matrix M from above, based on
        #  it create a matrix G such that X^T * G is the gradient.

        # ====== YOUR CODE: ======
        marginal_hinge_loss = self.grad_ctx['marginal_hinge_loss']
        y, x, rows = self.grad_ctx['y'], self.grad_ctx['x'], self.grad_ctx['rows']
        marginal_hinge_loss[marginal_hinge_loss > 0] = 1
        y_marginal_hinge_loss = torch.sum(marginal_hinge_loss, dim=1)
        marginal_hinge_loss[rows, y] = y_marginal_hinge_loss
        grad = torch.matmul(x.transpose(0, 1), marginal_hinge_loss)
        # ========================

        return grad
