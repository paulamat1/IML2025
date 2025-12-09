import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from math import inf
import os
from model_structure import SimpleCNN
from spectrogram_dataset import NpySpectrogramDataset
from tensorboard import SummaryWriter

IMG_H = 80
IMG_W = 300
BATCH_SIZE = 32
EPOCHS = 20

AUG_RATIO=1.0
USE_NOISE=True
USE_CHOP=True
USE_FAST=True
USE_SLOW=True

TRAIN_PATH = "/kaggle/input/spectrograms-1/spectograms/raw_data/train_data"
VALID_PATH = "/kaggle/input/spectrograms-1/spectograms/raw_data/validation_data"
TEST_PATH = "/kaggle/input/spectrograms-1/spectograms/raw_data/test_data"
    
    
def train_model(model, train_loader, device, criterion, optimizer):
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    model.train()
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device).long() #converts to int64
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        values, predictions = outputs.max(1)
        train_correct += (predictions == labels).sum().item() #.item() converts tensor sum to python int
        train_total += labels.size(0) 

    train_loss = train_loss / train_total
    train_acc = train_correct / train_total
    print(f"Train Loss: {train_loss:.4f},  Train Acc: {train_acc:.4f}")
    return train_loss, train_acc


def validate_model(model, valid_loader, device, criterion):
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad(): #Validation should NOT compute gradients, update weights
        for inputs, labels in valid_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).long()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            values, predictions = outputs.max(1)
            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        print(f"Validation Loss: {val_loss:.4f},  Validation Acc: {val_acc:.4f}")
        return val_loss, val_acc


def test_model(model, test_loader, device, criterion):
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).long()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * inputs.size(0)
            values, predictions = outputs.max(1)
            test_correct += (predictions == labels).sum().item()
            test_total += labels.size(0)

        test_loss /= test_total
        test_acc = test_correct / test_total
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        return test_loss, test_acc

def compute_f1(model, data_loader, device):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).long()

            outputs = model(inputs)
            values, predictions = outputs.max(1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    macro_f1 = f1_score(all_labels, all_predictions, average="macro")
    weighted_f1 = f1_score(all_labels, all_predictions, average="weighted")
    return macro_f1, weighted_f1


def main():    
    model_name = "simple_cnn" 
    #Initialize state saving
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "test_loss": [],
        "test_acc": [],
        "f1_macro": [],
        "f1_weighted": []
    }

    tensorboard_dir = os.path.join("tensor_board_states", model_name)
    writer = SummaryWriter(log_dir=tensorboard_dir)
    best_val_loss = inf

    checkpoint_dir = os.path.join("checkpoints", model_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")

    #Data Loading 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_dataset = NpySpectrogramDataset(TRAIN_PATH, mel_bins=IMG_H, expected_width=IMG_W, aug_ratio=AUG_RATIO, use_noise=USE_NOISE, use_fast=USE_FAST, use_slow=USE_SLOW, use_chop=USE_CHOP)
    valid_dataset = NpySpectrogramDataset(VALID_PATH, mel_bins=IMG_H, expected_width=train_dataset.expected_width ,class_to_label=train_dataset.class_to_label)
    test_dataset  = NpySpectrogramDataset(TEST_PATH, mel_bins=IMG_H, expected_width=train_dataset.expected_width ,class_to_label=train_dataset.class_to_label)

    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2, pin_memory = True)
    valid_loader = DataLoader(valid_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 2, pin_memory = True)
    test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 2, pin_memory = True)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-3) #model.parameters() = all weights

    #Train
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_model(model, train_loader, device, criterion, optimizer)

        history["train_loss"].append(train_loss)  
        history["train_acc"].append(train_acc)    
        writer.add_scalar("train_loss", train_loss, epoch+1)  
        writer.add_scalar("train_acc",  train_acc,  epoch+1) 

        #Validate
        val_loss, val_acc = validate_model(model, valid_loader, device, criterion)

        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        writer.add_scalar("val_loss", val_loss, epoch+1)
        writer.add_scalar("val_acc",  val_acc,  epoch+1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,                                 
                "model_state": model.state_dict(),              
                "optimizer_state": optimizer.state_dict(),     
                "val_loss": val_loss, 
                "val_acc": val_acc,
                "img_shape": (IMG_H, IMG_W),                    
                "class_to_label": train_dataset.class_to_label,     
            }, best_model_path)

    #Test
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    test_loss, test_acc = test_model(model, test_loader, device, criterion)

    history["test_loss"].append(test_loss)
    history["test_acc"].append(test_acc)
    writer.add_scalar("test_loss", test_loss, EPOCHS)
    writer.add_scalar("test_acc",  test_acc,  EPOCHS)

    macro_f1, weighted_f1 = compute_f1(model, train_loader, device)

    print(f"Macro averaged f1-score: {macro_f1}")
    print(f"Weighted f1-score", {weighted_f1})
    history["f1_macro"].append(macro_f1)
    history["f1_weighted"].append(weighted_f1)

    history_path = os.path.join(checkpoint_dir, "history.pt")
    torch.save(history, history_path)  
    writer.close()


if __name__ == "__main__":
    main()
