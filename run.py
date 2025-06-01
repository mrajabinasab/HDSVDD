import torch
import numpy as np
import pandas as pd
from models import *
from trainer import *
from utils import *
from sklearn.model_selection import KFold
import os
import pickle


def load_and_prepare_data(folder_path, n_splits=5, random_seed=42):
    normal_data = pd.read_csv(os.path.join(folder_path, 'normal.csv'), header=None, low_memory=False)
    outliers = pd.read_csv(os.path.join(folder_path, 'anomaly.csv'), header=None, low_memory=False)
    normal_data['outlier'] = 0
    outliers['outlier'] = 1
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    fold_data = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(normal_data)):
        train_data = normal_data.iloc[train_idx]
        test_data = normal_data.iloc[test_idx]
        test_inliers = test_data.shape[0]
        test_data = pd.concat([test_data, outliers])
        test_labels = np.array([0] * test_inliers + [1] * outliers.shape[0])
        train_data = train_data.drop(columns=['outlier'])
        test_data = test_data.drop(columns=['outlier'])
        fold_data.append({
            'fold': fold,
            'train_data': train_data.values,
            'test_data': test_data.values,
            'test_labels': test_labels
        })
    
    return fold_data

results = {}

dataset_base_path = 'datasets'
dataset_folders = [f for f in os.listdir(dataset_base_path) if os.path.isdir(os.path.join(dataset_base_path, f))]

print("Hybrid Magnifying Deep SVDD")
print('*'.center(80, '*'))

for dataset in dataset_folders:
    folder_path = os.path.join(dataset_base_path, dataset)
    print(f"\nProcessing dataset: {dataset}")
    
    fold_data = load_and_prepare_data(folder_path, n_splits=5, random_seed=42)
    
    dataset_results = {}
    
    for fold_info in fold_data:
        fold = fold_info['fold']
        train = fold_info['train_data']
        test = fold_info['test_data']
        true_labels = fold_info['test_labels']
        
        print(f"\nProcessing fold {fold + 1}/5 for dataset {dataset}")
        
        device = torch.device('cpu')
        print(f"Using device: {device}")
        pretraining_phase = True
        validation_phase = False
        learning_rate = 1e-3
        weight_decay = 1e-5
        learning_milestones = [500]
        pretraining_epochs = 50
        training_epochs = 250
        batch_size = 1024
        batch_size = min(batch_size, len(train))
        input_dim = train.shape[1]
        
        if input_dim < 4:
            raise Exception("input_dim must be at least 4.")
        
        ae_latent_dim = 1 if input_dim <= 2 else 3
        dsvdd_num_layers = 6 if input_dim > 8 else input_dim - 3
        dout_dim = 30 if input_dim >= 42 else input_dim - dsvdd_num_layers
        ae_extra_layers = 2
        granularity_diff = 2 if input_dim - dsvdd_num_layers > 2 else input_dim - dsvdd_num_layers - 1
        chunk_size = 8192
        alpha = 0.5
        beta = 0.5

        model = HybridDeepSVDD(dataset, learning_rate, weight_decay, learning_milestones, 
                              pretraining_epochs, training_epochs, batch_size, device, 
                              pretraining_phase, input_dim, ae_latent_dim, dout_dim, 
                              dsvdd_num_layers, ae_extra_layers, granularity_diff, 
                              chunk_size, validation_phase, train, train)
        
        print("Pretraining Model...")
        model.pretrain()
        print("Pretraining is Done!")
        print("Training Model...")
        model.maintrain()
        print("Training is Done!")
        
        model_scripted = torch.jit.script(model.net)
        model_scripted.save(f'mod_{dataset}_fold{fold + 1}.pt')
        model_scriptedb = torch.jit.script(model.netb)
        model_scriptedb.save(f'modb_{dataset}_fold{fold + 1}.pt')
        
        print("Projecting Test Data using The Model...")
        test_predat = chunkfeed(model.net, test, chunk_size, dout_dim).numpy()
        test_predbt = chunkfeed(model.netb, test, chunk_size, dout_dim).numpy()
        
        print("Calculating SVDD Loss for Test Data...")
        hdsvdd_losses = np.zeros([len(test), 2])
        for j in range(len(test)):
            hdsvdd_losses[j, 0] = svddloss(test_predat[j, :], model.c)
            hdsvdd_losses[j, 1] = svddloss(test_predbt[j, :], model.cb)
        
        dataset_results[f'fold_{fold + 1}'] = {
            'subspace': hdsvdd_losses,
            'true_labels': true_labels
        }
    
    results[dataset] = dataset_results

# Save results
with open('final_results_50_250_cv.pkl', 'wb') as file:
    pickle.dump(results, file)