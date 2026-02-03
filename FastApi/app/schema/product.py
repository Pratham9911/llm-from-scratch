from pydantic import BaseModel , Field # Field is used to add metadata and validation to model fields
from typing import Annotated , Optional , List
from datetime import datetime

# Define a Pydantic model for product creation
# it ensures that the incoming request data is validated and structured correctly

class Product(BaseModel):
    id:int=0
    name : Annotated[str,Field(
        min_length=6 ,
          max_length=30 ,
            title="Product Name" ,
              description="Name of the product" , 
              examples=["Laptop","SmartPhone"]
              )]
    price: Annotated[float,Field(
        gt=0.0,
        default=10.0,
        title="Product Price",
        description="Price of the product, must be greater than zero"

    )]
    stock:Annotated[int,Field(
        ge=0,
        title="Product Stock",
        description="Number of items in stock, must be zero or greater"
    )]
    category:Annotated[str,Field(
        min_length=3,
        max_length=20,
        title="Product Category",
        description="Category of the product"
    )]
    description:str="No description available"
    
    tags: Annotated[Optional[List[str] ],Field(
       default=None,
       max_length=10,
       description="List of tags "
   )] 
    
    created_at: datetime 


class ProductUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=6, max_length=30)
    price: Optional[float] = Field(None, gt=0.0)
    stock: Optional[int] = Field(None, ge=0)    
    category: Optional[str] = Field(None, min_length=3, max_length=20)
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    
  
