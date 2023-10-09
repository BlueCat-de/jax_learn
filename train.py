
import jax
import jax.numpy as jnp 
from clu import metrics
from flax.training import train_state  # Useful dataclass to keep train state
from flax import struct                # Flax dataclasses
import optax                           # Common loss functions and optimizers
from module import CNN
from dataset import get_datasets
import tensorflow as tf
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

  train_ds, test_ds = get_datasets(num_epochs, batch_size)

  tf.random.set_seed(0)
  init_rng = jax.random.key(0)

  learning_rate = 0.01
  momentum = 0.9
  cnn = CNN()

  state = create_train_state(cnn, init_rng, learning_rate, momentum)
  del init_rng  # Must not be used anymore.

  # since train_ds is replicated num_epochs times in get_datasets(), we divide by num_epochs
  num_steps_per_epoch = train_ds.cardinality().numpy() // num_epochs
  # import ipdb
  # ipdb.set_trace()

  # metrics_history = {'train_loss': [],
  #                  'train_accuracy': [],
  #                  'test_loss': [],
  #                  'test_accuracy': []}
  
  for step, batch in enumerate(train_ds.as_numpy_iterator()):

    # Run optimization steps over training batches and compute batch metrics
    state = train_step(state, batch) # get updated train state (which contains the updated parameters)
    state = compute_metrics(state=state, batch=batch) # aggregate batch metrics
    # ipdb.set_trace()

    wandb.log({
      'train_loss': state.metrics.compute()['loss'],
      'train_accuracy': state.metrics.compute()['accuracy'],
    })

    if (step + 1) % num_steps_per_epoch == 0: # one training epoch has passed
      # for metric, value in state.metrics.compute().items(): # compute metrics
      #   metrics_history[f'train_{metric}'].append(value) # record metrics
      state = state.replace(metrics=state.metrics.empty()) # reset train_metrics for next training epoch

      # Compute metrics on the test set after each training epoch
      test_state = state
      for test_batch in test_ds.as_numpy_iterator():
        test_state = compute_metrics(state=test_state, batch=test_batch)

      # for metric,value in test_state.metrics.compute().items():
      #   metrics_history[f'test_{metric}'].append(value)

  
      # print(f"test epoch: {(step+1) // num_steps_per_epoch}, "
      #       f"loss: {metrics_history['test_loss'][-1]}, "
      #       f"accuracy: {metrics_history['test_accuracy'][-1] * 100}")
      wandb.log({
        'test_loss': test_state.metrics.compute()['loss'],
        'test_accuracy': test_state.metrics.compute()['accuracy'],
      })



if __name__ == "__main__":
    main()