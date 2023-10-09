
import jax
import jax.numpy as jnp 
from clu import metrics
from flax.training import train_state  # Useful dataclass to keep train state
from flax import struct                # Flax dataclasses
import optax                           # Common loss functions and optimizers
from module import CNN
from dataset_torch import get_datasets
# import tensorflow as tf
import torch
import wandb
import ipdb

@struct.dataclass
class Metrics(metrics.Collection):
  accuracy: metrics.Accuracy
  loss: metrics.Average.from_output('loss')

class TrainState(train_state.TrainState):
  metrics: Metrics

def create_train_state(module, rng, learning_rate, momentum):
  """Creates an initial `TrainState`."""
  params = module.init(rng, jnp.ones([1, 28, 28, 1]))['params'] # initialize parameters by passing a template image
  tx = optax.sgd(learning_rate, momentum)
  return TrainState.create(
      apply_fn=module.apply, params=params, tx=tx,
      metrics=Metrics.empty())

@jax.jit
def train_step(state, batch):
  """Train for a single step."""
  # ipdb.set_trace()
  def loss_fn(params):
    logits = state.apply_fn({'params': params}, batch['image'])
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch['label']).mean()
    return loss
  grad_fn = jax.grad(loss_fn)
  grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  return state

@jax.jit
def compute_metrics(*, state, batch):
  logits = state.apply_fn({'params': state.params}, batch['image'])
  loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch['label']).mean()
  metric_updates = state.metrics.single_from_model_output(
    logits=logits, labels=batch['label'], loss=loss)
  metrics = state.metrics.merge(metric_updates)
  state = state.replace(metrics=metrics)
  return state

def main():

  wandb.login()

  

  num_epochs = 10
  batch_size = 32

  run = wandb.init(
    project='jax-learn',
    config = {
      'num_epochs': num_epochs,
      'batch_size': batch_size,
    }
  )

  train_loader, test_loader = get_datasets(batch_size)

  torch.random.manual_seed(0)
  init_rng = jax.random.key(0)

  learning_rate = 0.01
  momentum = 0.9
  cnn = CNN()

  state = create_train_state(cnn, init_rng, learning_rate, momentum)
  del init_rng  # Must not be used anymore.

  # # since train_ds is replicated num_epochs times in get_datasets(), we divide by num_epochs
  # num_steps_per_epoch = train_ds.cardinality().numpy() // num_epochs
  # import ipdb
  # ipdb.set_trace()

  # metrics_history = {'train_loss': [],
  #                  'train_accuracy': [],
  #                  'test_loss': [],
  #                  'test_accuracy': []}
  for epoch in range(num_epochs):
    for batch in train_loader:

      # ipdb.set_trace()
      # Run optimization steps over training batches and compute batch metrics
      batch = {
        'image': jnp.array(batch[0].view(32, 28, 28, 1)),
        'label': jnp.array(batch[1])
      }
      state = train_step(state, batch) # get updated train state (which contains the updated parameters)
      state = compute_metrics(state=state, batch=batch) # aggregate batch metrics
      # ipdb.set_trace()

      wandb.log({
        'train_loss': state.metrics.compute()['loss'],
        'train_accuracy': state.metrics.compute()['accuracy'],
      })

      
    test_state = state
    for test_batch in test_loader:
      test_batch = {
        'image': jnp.array(test_batch[0].view(32, 28, 28, 1)),
        'label': jnp.array(test_batch[1])
      }
      test_state = compute_metrics(state=test_state, batch=test_batch)

    wandb.log({
      'test_loss': test_state.metrics.compute()['loss'],
      'test_accuracy': test_state.metrics.compute()['accuracy'],
    })



if __name__ == "__main__":
    main()