"""
Costruction of dataloader for end-to-end architecture for CT reconstruction of the Mayodataset
images using TV images as targets.

Expected directory structure:
    data/
        fbp/           <- FBP reconstructions  (input)
            angle_180/
                img_000.npy
                ...
            angle_090/
            angle_060/
            angle_045/
        tv/            <- TV reconstructions   (target)
            angle_180/
            angle_090/
            angle_060/
            angle_045/

"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random

class TrainDataset(Dataset): 
    def __init__(self, fbp_files: list, tv_files: list, transform=None):
        self.fbp_files = fbp_files
        self.tv_files  = tv_files
        self.transform = transform
        
        assert len(self.fbp_files) == len(self.tv_files), "Mismatch numero file!"

    def __len__(self):
        return len(self.fbp_files)

    def __getitem__(self, idx):
        # Caricamento file binari (preserva i float32 della TV)
        fbp = np.load(self.fbp_files[idx]) # (256, 256)
        tv  = np.load(self.tv_files[idx])  # (256, 256)

        # Converti in tensori e aggiungi canale
        fbp = torch.from_numpy(fbp).float().unsqueeze(0)
        tv  = torch.from_numpy(tv).float().unsqueeze(0)

        # Data Augmentation (Esempio veloce: Random Horizontal Flip)
        if self.training: # Applica solo in fase di training
            if random.random() > 0.5:
                fbp = torch.flip(fbp, [2])
                tv  = torch.flip(tv, [2])

        return fbp, tv

def get_dataloaders_patient_wise(fbp_dir, tv_dir, batch_size=8, seed=42):
    fbp_all_files = sorted(glob.glob(os.path.join(fbp_dir, "*.npy")))
    tv_all_files = sorted(glob.glob(os.path.join(tv_dir, "*.npy")))
    
    # 2. Raggruppa i file per Paziente (Dizionario: { "L067": [file1, file2...], ... })
    patients_dict_fbp = {}
    patients_dict_tv = {}
    
    for fbp_path, tv_path in zip(fbp_all_files, tv_all_files):
        # Estrai il nome del file (es: "L067_slice001.npy")
        filename = os.path.basename(fbp_path)
        
        # LOGICA DA ADATTARE: Come estraiamo l'ID? 
        # Se i file sono separati da underscore, prendiamo la prima parte
        patient_id = filename.split('_')[0] 
        
        if patient_id not in patients_dict_fbp:
            patients_dict_fbp[patient_id] = []
            patients_dict_tv[patient_id] = []
            
        patients_dict_fbp[patient_id].append(fbp_path)
        patients_dict_tv[patient_id].append(tv_path)

    # 3. Lista dei pazienti univoci
    unique_patients = list(patients_dict_fbp.keys())
    
    # Mischia la lista dei pazienti in modo riproducibile
    random.seed(seed)
    random.shuffle(unique_patients)
    
    # 4. Calcola quanti PAZIENTI (non quanti file) vanno in ogni split
    n_patients = len(unique_patients)
    n_val = max(1, int(n_patients * 0.15)) # Es. 15% per Validation
    n_test = max(1, int(n_patients * 0.15)) # Es. 15% per Test
    n_train = n_patients - n_val - n_test
    
    # Seleziona i pazienti per i vari set
    train_patients = unique_patients[:n_train]
    val_patients = unique_patients[n_train:n_train + n_val]
    test_patients = unique_patients[-n_test:]
    
    print(f"Pazienti totali: {n_patients}")
    print(f"Train Pazienti: {len(train_patients)} | Val Pazienti: {len(val_patients)} | Test Pazienti: {len(test_patients)}")

    # 5. Riempi le liste di file finali assemblando i pazienti scelti
    train_fbp, train_tv = [], []
    val_fbp, val_tv = [], []
    test_fbp, test_tv = [], []
    
    for p in train_patients:
        train_fbp.extend(patients_dict_fbp[p])
        train_tv.extend(patients_dict_tv[p])
        
    for p in val_patients:
        val_fbp.extend(patients_dict_fbp[p])
        val_tv.extend(patients_dict_tv[p])
        
    for p in test_patients:
        test_fbp.extend(patients_dict_fbp[p])
        test_tv.extend(patients_dict_tv[p])

    # 6. Crea i Dataset e i Dataloader
    train_ds = TrainDataset(train_fbp, train_tv)
    val_ds = TrainDataset(val_fbp, val_tv)
    test_ds = TrainDataset(test_fbp, test_tv)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False) # Batch 1 per il test finale

    return train_loader, val_loader, test_loader
