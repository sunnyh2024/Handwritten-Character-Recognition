import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

# This was going to be the custom optimizer that we use for our CNN instead of usin to built-in SGD to
# learn more about how Keras optimizes work. We realized that we researched the wrong optimizer as 
# there is a v2 that our model uses which makes this incompatible. The documentation for v2 and the resources online
# were not that clear at all for this version so we gave up on it. Overall, our
# planning in the research phase for SGD should've been much better.

class SGDv2(keras.optimizers.Optimizer):
    def __init__(self, learning_rate=0.001, momentum=0.66, name="SGDv2", **kwargs):
        """Call super and _set_hyper methods to store hyperparamters."""
        super(SGDv2, self).__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get(learning_rate))
        self._set_hyper("decay", self._inital_decay)
        self._set_hyper("momentum", momentum)

    def _create_slot(self, var_list):
        """For momentum optimization, add a momentum slot per model var.
        Slots are optimizer variables that keras uses"""
        for var in var_list:
            self.add_slot(var, "momentum")

    @tf.function
    def _resource_apply_dense(self, grad, var):
        """Given the grad and var which are tensors that contain the values of the gradient and the
        variables, update the slots and perform one optimization step for one var """
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype) # handle learning rate decay
        momentum_var = self.get_slot(var, "momentum")
        momentum_hyper = self._get_hyper("momentum", var_dtype)
        new_momentum_var = momentum_var * momentum_hyper - (1. - momentum_hyper)* grad
        momentum_var.assign(new_momentum_var)
        var.assign_add(momentum_var * lr_t)

    def _resource_apply_sparse(self, grad, var):
        raise NotImplementedError

    def get_config(self):
   
        base_config = super().get_config()
        return {
            **base_config,
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "decay": self._serialize_hyperparameter("decay"),
            "momentum": self._serialize_hyperparameter("momentum"),
        }