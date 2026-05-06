"""
Costruction of dataloader for end-to-end architecture for CT reconstruction of the Mayo dataset
using TV images as targets.

Expected directory structure:
    data/
        train/
            sinograms/ 
                angle_090/
                    img_001.npy
                    ...
            tv/
                angle_090/
        val/
            sinograms/
            tv/
        test/
            sinograms/
            tv/
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random

class CTDataset(Dataset): 
    def __init__(self, input_dir: str, target_dir: str, is_train: bool = False):
        """
        Carica le coppie (input, target) leggendo tutti i file .npy nelle cartelle specificate.
        is_train: se True, applica data augmentation per ridurre l'overfitting.
        """
        self.input_files = sorted(glob.glob(os.path.join(input_dir, "*.npy")))
        self.target_files = sorted(glob.glob(os.path.join(target_dir, "*.npy")))
        self.is_train = is_train
        
        # Controlli di sicurezza
        assert len(self.input_files) == len(self.target_files), \
            f"Mismatch: trovati {len(self.input_files)} input e {len(self.target_files)} target in {input_dir}"
        assert len(self.input_files) > 0, f"Nessun file .npy trovato in {input_dir}!"

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        # Caricamento file binari (preserva i float32 essenziali per la TV)
        x = np.load(self.input_files[idx]).astype(np.float32)
        y = np.load(self.target_files[idx]).astype(np.float32)

        # Converti in tensori e aggiungi la dimensione del canale (1, 256, 256)
        x = torch.from_numpy(x).unsqueeze(0)
        y = torch.from_numpy(y).unsqueeze(0)

        # Data Augmentation (solo durante il training)
        if self.is_train: 
            # 50% di probabilità di flip orizzontale
            if random.random() > 0.5:
                x = torch.flip(x, [2])
                y = torch.flip(y, [2])
                
            # Puoi aggiungere anche il flip verticale per le CT, ha senso anatomicamente!
            if random.random() > 0.5:
                x = torch.flip(x, [1])
                y = torch.flip(y, [1])

        return x, y


def get_dataloaders(base_data_dir: str, angle: str, batch_size: int = 8):
    """
    Costruisce e restituisce direttamente i tre Dataloader.
    base_data_dir: la cartella radice (es. 'data')
    angle: l'angolo sotto forma di stringa (es. '090')
    """
    
    # 1. Costruisci i percorsi esatti per TRAIN
    train_sino = os.path.join(base_data_dir, "train", "sinograms", f"angle_{angle}")
    train_tv   = os.path.join(base_data_dir, "train", "tv", f"angle_{angle}")
    
    # 2. Costruisci i percorsi esatti per VAL
    val_sino = os.path.join(base_data_dir, "val", "sinograms", f"angle_{angle}")
    val_tv   = os.path.join(base_data_dir, "val", "tv", f"angle_{angle}")
    
    # 3. Costruisci i percorsi esatti per TEST
    test_sino = os.path.join(base_data_dir, "test", "sinograms", f"angle_{angle}")
    test_tv   = os.path.join(base_data_dir, "test", "tv", f"angle_{angle}")

    # 4. Inizializza i Dataset
    # NOTA: is_train=True SOLO per il training set!
    train_ds = CTDataset(train_sino, train_tv, is_train=True)
    val_ds   = CTDataset(val_sino, val_tv, is_train=False)
    test_ds  = CTDataset(test_sino, test_tv, is_train=False)

    print(f"Dataset caricato (Angolo {angle}) -> Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # 5. Crea i DataLoader
    # pin_memory=True velocizza il passaggio dei dati CPU -> GPU
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    # Il test loader di solito ha batch_size=1 per fare calcoli più precisi in fase di inferenza
    test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False, pin_memory=True) 

    return train_loader, val_loader, test_loader
