import os
from dotenv import load_dotenv
from woocommerce import API

def fetch_recent_products(wcapi, limit=5):
    """Fetch the most recent products from WooCommerce."""
    response = wcapi.get("products", params={"orderby": "date", "order": "desc", "per_page": limit})

    if response.status_code == 200:
        products = response.json()
        result = []
        for product in products:
            first_image = (
                product['images'][0]['src']
                if product.get('images') and len(product['images']) > 0
                else "No image available"
            )
            text_representation = (
                f"{product['name']}. Brand: {product.get('brand', 'Unknown')}. "
                f"Description: {product.get('description', '')}. "
                f"Tags: {', '.join([tag['name'] for tag in product.get('tags', [])])}."
            )
            product_info = {
                "id": product['id'],
                "text": product['name'],
                "image": first_image
            }
            result.append(product_info)
        return result
    else:
        print(f"Failed to fetch products: {response.status_code}, {response.text}")
        return []
