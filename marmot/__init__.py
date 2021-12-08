import os
import click
import json
import random
import pandas as pd
import numpy as np
from importlib import import_module


def import_object(path):
    *module, obj = path.split('.')
    module = '.'.join(module)
    return getattr(import_module(module), obj)


def build_pool(features_dict, keys_without_target):
    key_pool = []
    features_pool = []
    for key, features in features_dict.items():
        if key in keys_without_target:
            key_pool.append(key)
            features_pool.append(features)
    return key_pool, np.array(features_pool)


def marmot(config):
    # create our broker
    broker = import_object(config['broker'])(**config['config'])
    
    # build/load our features
    print('Building features...')
    if not os.path.exists(config['config']['features_dir']):
        os.mkdir(config['config']['features_dir'])
    feature_builders = [
        import_object(feature['path'])(override=feature.get('override', False), **config['config'])
        for feature in config['features']
    ]
    features_dfs = []
    for feature_builder in feature_builders:
        print(f'{feature_builder.__class__.__name__}...')
        feature_builder.build_features(broker.read_and_build_keys())
        features_dfs.append(pd.read_csv(feature_builder.file_path))
    
    print('Merging features...')
    features_df = features_dfs[0]
    for extra_df in features_dfs[1:]:
        features_df = features_df.merge(extra_df, on='key', how='inner')
    feature_columns = sorted([c for c in features_df.columns if c != 'key'])
    features = {
        r['key']: np.array([r[c] for c in feature_columns])
        for _, r in features_df.iterrows()
    }
    
    # load the targets
    print('Loading targets...')
    if not os.path.exists(config['config']['targets_dir']):
        os.mkdir(config['config']['targets_dir'])
    training_targets_path = os.path.join(config['config']['targets_dir'], config['config']['target_id'] + '_train.csv')
    if os.path.exists(training_targets_path):
        training_targets = {
            r['key']: r['target']
            for _, r in pd.read_csv(training_targets_path).iterrows()
        }
    else:
        training_targets = {}
    testing_targets_path = os.path.join(config['config']['targets_dir'], config['config']['target_id'] + '_test.csv')
    if os.path.exists(testing_targets_path):
        testing_targets = {
            r['key']: r['target']
            for _, r in pd.read_csv(testing_targets_path).iterrows()
        }
    else:
        testing_targets = {}

    # build our interface
    interface = import_object(config['interface'])(broker=broker, **config['config'])

    # ensure we have enough data to train on
    print('Gathering training data...')
    keys_without_target = set(features_df['key']) - set(training_targets) - set(testing_targets)
    new_test_data = False
    new_train_data = False
    while (
        len(training_targets) < config['config']['minimum_to_train'] 
        or len(set(training_targets.values())) < 2
    ):
        key = random.sample(keys_without_target, 1)[0]
        target = interface.get_target(key)
        if random.random() < config['config']['test_train_split']:
            testing_targets[key] = target
            new_test_data |= True
        else:
            training_targets[key] = target
            new_train_data |= True
        keys_without_target.remove(key)
    
    # save the new targets
    if new_train_data:
        pd.DataFrame(
            [{'key': key, 'target': target} for key, target in training_targets.items()]
        ).to_csv(training_targets_path, index=False)
    if new_test_data:
        pd.DataFrame(
            [{'key': key, 'target': target} for key, target in testing_targets.items()]
        ).to_csv(testing_targets_path, index=False)

    # prepare the data for the learner
    print('Building learner...')
    training_keys = list(training_targets.keys())
    testing_keys = list(testing_targets.keys())
    X_training = np.array([
        features[key]
        for key in training_keys
    ])
    X_testing = np.array([
        features[key]
        for key in testing_keys
    ])
    y_training = np.array([
        training_targets[key]
        for key in training_keys
    ])
    y_testing = np.array([
        testing_targets[key]
        for key in testing_keys
    ])
    key_pool, X_pool = build_pool(features, keys_without_target)

    # build and train our learner
    learner = import_object(config['learner']['factory'])(
        X_training=X_training, 
        y_training=y_training, 
        query_method=config['learner']['query_method'],
        **config['config']
    )

    print('Active learning...')
    step = 1
    while len(X_pool):
        print('step:', step)
        step += 1
        # query and label
        queried_indices, _ = learner.query(X_pool, n_instances=config['config']['queries_per_round'])
        new_test_data = False
        new_train_data = False
        X_new_test = []
        y_new_test = []
        X_new_train = []
        y_new_train = []
        for index in queried_indices:
            key = key_pool[index]
            target = interface.get_target(key)
            if random.random() < config['config']['test_train_split']:
                testing_targets[key] = target
                y_new_test.append(target)
                X_new_test.append(features[key])
                new_test_data |= True
            else:
                training_targets[key] = target
                y_new_train.append(target)
                X_new_train.append(features[key])
                new_train_data |= True
            keys_without_target.remove(key)
        X_testing = np.concatenate([X_testing, np.array(X_new_test)])
        y_testing = np.concatenate([y_testing, np.array(y_new_test)])
        X_training = np.concatenate([X_training, np.array(X_new_train)])
        y_training = np.concatenate([y_training, np.array(y_new_train)])
        # reset the pool
        key_pool, X_pool = build_pool(features, keys_without_target)
        # teach
        if new_train_data:
            learner.teach(np.array(X_new_train), np.array(y_new_train))
        # save new targets
        if new_train_data:
            pd.DataFrame(
                [{'key': key, 'target': target} for key, target in training_targets.items()]
            ).to_csv(training_targets_path, index=False)
        if new_test_data:
            pd.DataFrame(
                [{'key': key, 'target': target} for key, target in testing_targets.items()]
        ).to_csv(testing_targets_path, index=False)
        # report scores
        print(len(training_targets))
        print(len(testing_targets))
        print('Training - samples:', y_training.shape[0], 'score:', learner.score(X_training, y_training))
        print('Testing - samples:', y_testing.shape[0], 'score:', learner.score(X_testing, y_testing))


@click.command()
@click.option('--config', required=True)
def cli(config):
    with open(config, 'r') as fh:
        config = json.load(fh)
    marmot(config)
