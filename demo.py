# General imports.
import os
import jax
import jax.numpy as jnp
import flax

# Huggingface datasets and transformers libraries.
import datasets
from transformers import BertTokenizerFast

# flax_bert-specific imports.
from flax import optim
import data
import modeling as flax_models
import training
from demo_lib import get_config, get_validation_splits, get_prefix, import_pretrained_params, create_model, create_optimizer, get_num_train_steps, get_learning_rate_fn

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

train_settings = {
    'train_batch_size': 32,
    'eval_batch_size': 8,
    'learning_rate': 5e-5,
    'num_train_epochs': 3,
    'dataset_path': 'glue',
    'dataset_name': 'mrpc'  # ['cola', 'mrpc', 'sst2', 'stsb', 'qnli', 'rte']
}

# Load the GLUE task.
dataset = datasets.load_dataset('glue', train_settings['dataset_name'])

# Get pre-trained config and update it with the train configuration.
config = get_config('bert-base-uncased', dataset)
config.update(train_settings)

# Load HuggingFace tokenizer and data pipeline.
tokenizer = BertTokenizerFast.from_pretrained(config.tokenizer)
data_pipeline = data.ClassificationDataPipeline(dataset, tokenizer)

# Create Flax model and optimizer.
pretrained_params = import_pretrained_params(config)
model = create_model(config, pretrained_params)
optimizer = create_optimizer(config, model, pretrained_params)

# Setup tokenizer, train step function and train iterator.
tokenizer.model_max_length = config.max_seq_length
num_train_steps = get_num_train_steps(config, data_pipeline)

learning_rate_fn = get_learning_rate_fn(config, num_train_steps)
train_history = training.TrainStateHistory(learning_rate_fn)
train_state = train_history.initial_state()

train_step_fn = training.create_train_step(clip_grad_norm=1.0)
train_iter = data_pipeline.get_inputs(
  split='train', batch_size=config.train_batch_size, training=True)

# Run training.
print(f'\nStarting training on {config.dataset_name} for {num_train_steps} '
      f'steps ({config.num_train_epochs:.0f} epochs)...\n')

for step, batch in zip(range(0, num_train_steps), train_iter):
  optimizer, train_state = train_step_fn(optimizer, batch, train_state)
  if step % 10 == 0:
    print(f'step {step}/{num_train_steps}')

print('\nTraining finished! Running eval...\n')

# Run eval.
eval_step = training.create_eval_fn()

for split in get_validation_splits(config.dataset_name):
  eval_iter = data_pipeline.get_inputs(
      split='validation', batch_size=config.eval_batch_size, training=False)
  eval_stats = eval_step(optimizer, eval_iter)
  eval_metric = datasets.load_metric(config.dataset_path, config.dataset_name)
  eval_metric.add_batch(
    predictions=eval_stats['prediction'],
    references=eval_stats['label'])
  eval_metrics = eval_metric.compute()
  for name, val in sorted(eval_metrics.items()):
    print(f'{get_prefix(split)}_{name} = {val:.06f}', flush=True)
