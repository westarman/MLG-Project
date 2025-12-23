"""
import torch 
import pandas as pd
import sqlite3 as sql

def load_forest_hetero_data(db_path):
    print(f"Connecting to database at {db_path}...")
    db = sql.connect(db_path)
    df_sestoj = pd.read_sql_query("SELECT etigl, etlst from sestoji_attr", db)
    db.close()
    return df_sestoj 


best_loss = float('inf')
best_pair = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
db_file = "../../datasets/forest_db.sqlite"
data_df = load_forest_hetero_data(db_file)
y_tensor = torch.tensor(data_df[['etigl', 'etlst']].values, dtype=torch.float32, device=device)
print("sm kle")
for int1 in range(8863):
    for int2 in range(13014):
        predict = torch.zeros((347338, 2), device=device)
        predict[:, 0] = int1
        predict[:, 1] = int2
        loss = torch.mean(torch.abs(predict - y_tensor[:347338]))
        if loss < best_loss:
            best_loss = loss
            best_pair = predict


print(f"Best pair of increments: {best_pair} with L1 loss: {best_loss}")
# Best pair of increments: tensor([[19., 16.],
#  with L1 loss: 206.38653564453125

Prepočasi --> lahko 

""" 
import torch
import pandas as pd
import sqlite3 as sql

def load_forest_hetero_data(db_path):
    print(f"Connecting to database at {db_path}...")
    db = sql.connect(db_path)
    df_sestoj = pd.read_sql_query("SELECT etigl, etlst FROM sestoji_attr", db)
    db.close()
    return df_sestoj

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
db_file = "../../datasets/forest_db.sqlite"
data_df = load_forest_hetero_data(db_file)

y_tensor = torch.tensor(data_df[['etigl', 'etlst']].values, dtype=torch.float32, device=device)
y_tensor = y_tensor[:347338] 

best_pair = torch.median(y_tensor, dim=0).values
best_loss = torch.mean(torch.abs(y_tensor - best_pair))

print(f"Best pair: {best_pair.tolist()} with L1 loss: {best_loss.item():.4f}")
# Connecting to database at ../../datasets/forest_db.sqlite...
# Best pair: [10.0, 25.0] with L1 loss: 100.3961
