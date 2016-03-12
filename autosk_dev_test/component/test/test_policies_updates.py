# -*- encoding: utf-8 -*-

import unittest
import os
import theano
import theano.tensor as T
import numpy as np


class PoliciesUpdatesTest(unittest.TestCase):
    learning_rate = 0.1
    gamma = 0.9
    epoch_step = 4
    power = 0.7
    shared_lr = theano.shared(learning_rate,
                              name='lr', strict=True)

    def policy_function(self, lr_policy, n_epoch):
        lr_base = self.learning_rate
        if lr_policy == "inv":
            decay = np.power((1 + self.gamma * n_epoch), (-self.power))
        elif lr_policy == "exp":
            decay = np.power(self.gamma, n_epoch)
        elif lr_policy == "step":
            if (n_epoch % self.epoch_step) == 0.0:
                decay = np.power(self.gamma, (np.floor(n_epoch / float(self.epoch_step))))
            else:
                decay = 1
        elif lr_policy == "fixed":
            decay = 1

        self.learning_rate = lr_base * decay

    def policy_update(self, lr_policy):
        epoch, gm, powr, step = T.scalars('epoch', 'gm', 'powr', 'step')
        if lr_policy == 'inv':
            decay = T.power(1+gm*epoch, -powr)
        elif lr_policy == 'exp':
            decay = gm ** epoch
        elif lr_policy == 'step':
            decay = T.switch(T.eq(T.mod_check(epoch, step), 0),
                             T.power(gm, T.floor_div(epoch, step)),
                             1.0)
        elif lr_policy == 'fixed':
            decay = T.constant(1.0, name='fixed', dtype='float32')

        return theano.function([gm, epoch, powr, step],
                               decay,
                               updates=[(self.shared_lr,
                                         self.shared_lr * decay)],
                               on_unused_input='ignore')

    def test_policy(self, policy='step'):
        for epoch in range(20):
            self.policy_function(policy, epoch+1)
        lr_numpy_updated = self.learning_rate

        update_function = self.policy_update(policy)
        for epoch in range(20):
            update_function(self.gamma, epoch+1, self.power, self.epoch_step)

        lr_theano_updated = self.shared_lr.get_value()

        print('Theano lr value is: {}'.format(lr_theano_updated))
        print('Numpy lr value is: {}'.format(lr_numpy_updated))
        self.assertAlmostEqual(lr_theano_updated, lr_numpy_updated)

    def test_all_policies(self):
        all_policies = ["fixed", "inv", "exp", "step"]
        for policy in all_policies:
            print("==== Values for {0} policy ====".format(policy))
            self.test_policy(policy)
            self.learning_rate = 0.1
            self.shared_lr.set_value(np.array(0.1).astype(np.float64))
