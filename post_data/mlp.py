#!/usr/bin/env python3


from neo4j import GraphDatabase, basic_auth, Driver
from dotenv import load_dotenv
import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import nn
import torch
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, f1_score
import numpy as np


class MLP(nn.Module):
    def __init__(self, input_size: int):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


def build_df(driver: Driver) -> pd.DataFrame:
    with driver.session() as session:
        query = """
            MATCH (n:Artist)
            WITH COLLECT(DISTINCT n.main_tag) AS all_tags
            RETURN all_tags;
        """
        all_tags = session.run(query).data()[0]["all_tags"]
        query = f"""
            MATCH (n:Artist {{in_last_fm: true}})
            WITH n
            ORDER BY n.popularity_scaled DESC
            LIMIT 1000
            WITH collect(n) AS artists
            UNWIND artists AS a0
            UNWIND artists AS a1
            WITH a0, a1, exists((a0)-[:COLLAB_WITH]-(a1)) AS edge_exists
            RETURN 
                a0.popularity_scaled AS a0_popularity_scaled,
                COALESCE(a0.begin_date, 0) AS a0_begin_date,
                COALESCE(a0.end_date, 0) AS a0_end_date,
                a0.ended AS a0_ended,
                a0.gender_1 AS a0_gender1,
                a0.gender_2 AS a0_gender2,
                a0.gender_3 AS a0_gender3,
                a0.gender_4 AS a0_gender4,
                a0.gender_5 AS a0_gender5,
                {",\n".join(f"CASE a0.main_tag WHEN \"{tag}\" THEN 1 ELSE 0 END AS a0_main_tag_{"_".join(tag.replace("/", "").replace("-", "").split())}" for tag in all_tags)},
                a0.type_1 AS a0_type1,
                a0.type_2 AS a0_type2,
                a0.type_3 AS a0_type3,
                a0.type_4 AS a0_type4,
                a0.type_5 AS a0_type5,
                a0.type_6 AS a0_type6,

                a1.popularity_scaled AS a1_popularity_scaled,
                COALESCE(a1.begin_date, 0) AS a1_begin_date,
                COALESCE(a1.end_date, 0) AS a1_end_date,
                a1.ended AS a1_ended,
                a1.gender_1 AS a1_gender1,
                a1.gender_2 AS a1_gender2,
                a1.gender_3 AS a1_gender3,
                a1.gender_4 AS a1_gender4,
                a1.gender_5 AS a1_gender5,
                {",\n".join(f"CASE a1.main_tag WHEN \"{tag}\" THEN 1 ELSE 0 END AS a1_main_tag_{"_".join(tag.replace("/", "").replace("-", "").split())}" for tag in all_tags)},
                a1.type_1 AS a1_type1,
                a1.type_2 AS a1_type2,
                a1.type_3 AS a1_type3,
                a1.type_4 AS a1_type4,
                a1.type_5 AS a1_type5,
                a1.type_6 AS a1_type6,
                edge_exists
            ;
        """
        result = session.run(query)  # type: ignore
        return pd.DataFrame(result.data())


def get_train_val_test_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test  # type: ignore

def train_model(model, criterion, optimizer, train_loader, val_loader, epochs=20):
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

        logging.info(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

def evaluate_model(model, test_loader):
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch).squeeze().cpu().numpy()
            y_true.extend(y_batch.numpy())
            y_pred.extend((outputs > 0.5).astype(int))

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    logging.info(f"Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, AUC: {roc_auc:.4f}")


def main(driver: Driver) -> None:
    logging.info("Gathering data...")
    df = build_df(driver).astype(np.float32)
    X_train, X_val, X_test, y_train, y_val, y_test = get_train_val_test_split(df)

    logging.info("Data gathered")

    train_loader = torch.utils.data.DataLoader(
        list(zip(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))), batch_size=64, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        list(zip(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))), batch_size=64, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        list(zip(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))), batch_size=64, shuffle=False
    )

    logging.info(X_train.shape[1])
    model = MLP(X_train.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    logging.info("Training...")

    train_model(model, criterion, optimizer, train_loader, val_loader, epochs=20)
    evaluate_model(model, test_loader)


if __name__ == '__main__':
    # Logging setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # .env read
    load_dotenv()
    DB_HOST = os.getenv("NEO4J_HOST")
    DB_PORT = os.getenv("NEO4J_PORT")
    DB_USER = os.getenv("NEO4J_USER")
    DB_PASS = os.getenv("NEO4J_PASS")

    # .env validation
    assert DB_HOST is not None and \
        DB_PORT is not None and \
        DB_USER is not None and \
        DB_PASS is not None, \
        "INVALID .env"

    # db connection
    driver = GraphDatabase.driver(f"bolt://{DB_HOST}:{DB_PORT}", auth=basic_auth(DB_USER, DB_PASS))

    assert torch.cuda.is_available()
    device = torch.device("cuda")

    main(driver)

    logging.info("DONE!")
