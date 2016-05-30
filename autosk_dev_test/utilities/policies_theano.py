# -*- encoding: utf-8 -*-
"""
Policies modules
"""
__author__ = 'mendozah'

import theano
import theano.tensor as T


initial_lr = 0.1
value_gamma = 0.8
value_epoch = 3.0
value_power = 0.75

lr = theano.shared(initial_lr, 'lr')
epoch, gamma, powr, step = T.scalars('epoch', 'gm', 'powr', 'step')
lr_policy = 'fixed'

if lr_policy == 'inv':
    decay = T.power(1 + gamma * epoch, -powr)
elif lr_policy == 'exp':
    decay = gamma ** epoch
elif lr_policy == 'step':
    decay = T.switch(T.eq(T.mod_check(epoch, step), 0),
                     T.power(gamma, T.floor_div(epoch, step)),
                     1.0)
elif lr_policy == 'fixed':
    decay = T.constant(1.0, name='fixed', dtype='float32')

policy_update = theano.function([gamma, epoch, powr],
                                decay,
                                updates=[(lr, lr * decay)],
                                on_unused_input='warn')

policy_update(value_gamma, value_epoch, value_power)
