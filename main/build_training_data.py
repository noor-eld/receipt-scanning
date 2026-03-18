# build_training_data.py
# Combines THREE sources into one training dataset:
#
# 1. Our original training data — receipt-style food names + non-food household items
# 2. Grocery Inventory dataset (Kaggle - willianoliveiragibin) — clean food product names
# 3. Groceries dataset (Kaggle - heeraldedhia) — grocery items including some non-food
#
# WHY ALL THREE?
# - Our original data has receipt-style names ("BANANA CAVENDISH", "GV PNT BUTTR")
# - Kaggle data adds variety and real-world product names
# - Together they make the model robust to both clean AND messy input

import pandas as pd

# --- SOURCE 1: Our original training data ---
print("Loading original training data...")
df_original = pd.read_csv("original_training_data.csv")
print(f"  {len(df_original)} entries ({len(df_original[df_original['label'] == 'food'])} food, {len(df_original[df_original['label'] == 'non_food'])} non_food)")

# --- SOURCE 2: Grocery Inventory (all food) ---
print("Loading Grocery Inventory dataset...")
df_inventory = pd.read_csv("Grocery_Inventory_new_v1.csv")
food_items_inv = df_inventory["Product_Name"].dropna().unique().tolist()
print(f"  {len(food_items_inv)} unique products")

# --- SOURCE 3: Groceries dataset (mixed) ---
print("Loading Groceries dataset...")
df_groceries = pd.read_csv("Groceries_dataset.csv")
grocery_items = df_groceries["itemDescription"].dropna().str.strip().unique().tolist()
grocery_items = [item for item in grocery_items if item != "itemDescription"]
print(f"  {len(grocery_items)} unique items")

# Non-food items from the Groceries dataset
DATASET_NON_FOOD = {
    "abrasive cleaner", "baby cosmetics", "bags", "bathroom cleaner",
    "candles", "cat food", "cleaner", "cling film/bags", "cookware",
    "decalcifier", "dental care", "detergent", "dish cleaner", "dishes",
    "dog food", "female sanitary products", "flower (seeds)",
    "flower soil/fertilizer", "hair spray", "house keeping products",
    "hygiene articles", "kitchen towels", "kitchen utensil", "light bulbs",
    "make up remover", "male cosmetics", "napkins", "newspapers",
    "pet care", "photo/film", "pot plants", "rubbing alcohol",
    "shopping bags", "skin care", "soap", "softener", "toilet cleaner",
}

# --- COMBINE EVERYTHING ---
print("\nBuilding combined training data...")

rows = []

# Add all original training data (this preserves receipt-style names + noise)
for _, row in df_original.iterrows():
    rows.append({"product_name": row["product_name"], "label": row["label"]})

# Add Grocery Inventory items (all food)
for item in food_items_inv:
    rows.append({"product_name": item, "label": "food"})

# Add Groceries dataset items (labeled)
for item in grocery_items:
    if item.lower().strip() in DATASET_NON_FOOD:
        rows.append({"product_name": item, "label": "non_food"})
    else:
        rows.append({"product_name": item, "label": "food"})

# Deduplicate
df = pd.DataFrame(rows)
df = df.drop_duplicates(subset=["product_name"])

food_count = len(df[df["label"] == "food"])
non_food_count = len(df[df["label"] == "non_food"])
print(f"  Food items: {food_count}")
print(f"  Non-food items: {non_food_count}")
print(f"  Total: {len(df)}")

df.to_csv("training_data.csv", index=False)
print(f"\nSaved to training_data.csv")
print("Now run: python train_model.py")