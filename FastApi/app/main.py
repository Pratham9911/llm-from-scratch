from fastapi import FastAPI , HTTPException , Query
# HttpExecption is used to handle errors
from services.products import get_all_products , add_product , remove_product , change_product
from schema.product import Product , ProductUpdate
from datetime import datetime
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
        default = False, 
        description = "Sort products by price in ascending order"
    ),
    order:str = Query(
        default = 'asc',
        description = "Order of sorting: 'asc' for ascending, 'desc' for descending when sort_by_price is True",
    ),
    limit:int = Query(
         default = 10,
         description = "Number of products to return",
         
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
    products = products[0:limit]
    return {
             'total': total,
             'products':products
         }

@app.get("/products/{id}")
def get_product_by_id(id:int):
    products = get_all_products()
    for product in products:
        if product["id"] == id:
            return product
    raise HTTPException(status_code=404, detail=f"Product with id {id} not found")


@app.post("/products")
def create_product(product:Product):
    
    product_dict = product.dict()
    product_dict['created_at'] = datetime.now().isoformat()
    try:
        add_product(product_dict)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return product_dict


@app.delete("/products/{id}")
def delete_product(product_id:int):
    try:
      res = remove_product(product_id)
      return res
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))   

@app.put("/products/{id}")
def update_product(id:int, update_data:ProductUpdate):
    update_product = update_data.dict(exclude_unset = True)
    try:
        updated_product = change_product(id, update_product)
        return updated_product
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e)) 
    