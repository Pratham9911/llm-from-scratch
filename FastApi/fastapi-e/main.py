from fastapi import FastAPI , HTTPException , Query
# HttpExecption is used to handle errors
from services.products import get_all_products

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello, World!"}

# @app.get("/product")
# def get_product():
#     return get_all_products()
# Query is used for parameter like product?name=abc , to access the name parameter

# search http://127.0.0.1:8000/products?name=Laptop&sort_by_price=false&order=asc to search products by name
@app.get("/products")
def list_products(
    name: str = Query(
       default = None,
       min_length = 1,
       max_length = 50,
       title = "Product Name",
       description = "Filter products by name"
    )
    ,
    sort_by_price:bool = Query(
        defaiult = False, 
        description = "Sort products by price in ascending order"
    ),
    order:str = Query(
        default = 'asc',
        description = "Order of sorting: 'asc' for ascending, 'desc' for descending when sort_by_price is True",
    )
):
    if name:
        clean_name = name.strip().lower()
        products = []
        for product in get_all_products():
            if clean_name in product["name"].lower():
                products.append(product)
    if not products:
            raise HTTPException(status_code = 404 , detail = f"No products found with name {name}")
    if sort_by_price:
        desc = order == 'desc'
        products = sorted(products, key=lambda x: x['price'] , reverse=desc )
    
    total = len(products)
    return {
             'total': total,
             'products':products
         }
