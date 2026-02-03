import json
from pathlib import Path
from typing import List , Dict

DATA_FILE = Path(__file__).parent.parent / "data" / "dummy.json"

def load_products() -> List[Dict]:

    if not DATA_FILE.exists():
        return []
    
    with open(DATA_FILE,"r",encoding="utf-8") as file:
        return json.load(file)
    
def get_all_products() -> List[Dict]:
    return load_products()

def save_products(products:List[Dict]) -> None:
    with open(DATA_FILE , "w",encoding="utf-8") as file:
        json.dump(products , file ,indent=4 , ensure_ascii=False)

def add_product(product:Dict) -> Dict:
    products = load_products()
    for p in products:
        if product['id'] == p['id'] :
            raise ValueError(f"Product with id {product['id']} already exists." )
    products.append(product)
    save_products(products)
    return product

def remove_product(id:int) -> str:
    products = load_products()
    for idx , p in enumerate(products):
        if p['id'] == id:
            deleted = products.pop(idx)
            save_products(products)
            return {"Deleted ": deleted}
    raise ValueError(f"Product with id {id} not found.")

def change_product(id:int, update_data:Dict):
    products = load_products()
    for idx , p in enumerate(products):
        if p['id'] == id:
            for key , value in update_data.items():
                if value is not None:
                    p[key] = value
            products[idx] = p
            save_products(products)
            return p
    raise ValueError(f"Product with id {id} not found.")
    
