# import os
# from dotenv import load_dotenv
# from woocommerce import API
# from fetch_wc_products import fetch_recent_products
# from woocommerce_encoder import WooCommerceEncoder
# from PIL import Image
# import requests
# from io import BytesIO
# import numpy as np


# # Load environment variables from .env file
# load_dotenv()

# # Retrieve WooCommerce API keys from environment
# consumer_key = os.getenv("WOOCOMMERCE_CONSUMER_KEY")
# consumer_secret = os.getenv("WOOCOMMERCE_CONSUMER_SECRET")

# # Initialize WooCommerce API
# wcapi = API(
#     url="https://streetmarkets.fr/app/",  # Your WooCommerce store URL
#     consumer_key=consumer_key,
#     consumer_secret=consumer_secret,
#     version="wc/v3"
# )

# def download_image_as_pil(url):
#     response = requests.get(url)
#     response.raise_for_status()
#     return Image.open(BytesIO(response.content))

# if __name__ == "__main__":
#     try:
#         # Fetch recent products
#         products = fetch_recent_products(wcapi, limit=5)
#         print(products)
#         # Initialize encoder
#         encoder = WooCommerceEncoder()

#         # Prepare embeddings
#         for product in products:
#             image_url = product.get("image")
#             text = product.get("text", "")

#             try:
#                 # Download image
#                 image = download_image_as_pil(image_url)
#                 if not isinstance(image, Image.Image):
#                     raise ValueError("Invalid image format")

#                 # Generate embeddings
#                 image_embedding = encoder.encode_images([image])[0]
#                 text_embedding = encoder.encode_text([text])[0]

#                 # Combine text and image embeddings
#                 combined_embedding = np.concatenate((text_embedding, image_embedding))
#                 combined_embedding = combined_embedding / np.linalg.norm(combined_embedding)

#                 print(f"Product ID: {product['id']}")
#                 print(f"Image Embedding: {image_embedding[:5]}...")  # Display first 5 values
#                 print(f"Text Embedding: {text_embedding[:5]}...")  # Display first 5 values
#                 print(f"Combined Embedding: {combined_embedding[:5]}...")  # Display first 5 values

#             except Exception as e:
#                 print(f"Error processing product ID {product['id']}: {e}")

#     except Exception as e:
#         print(f"Failed to execute embedding: {e}")


import uuid
import tqdm
import requests
from PIL import Image
from encoder import FashionCLIPEncoder

# Constants
BATCH_SIZE = 4
REQUESTS_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
IMAGE_URLS = [
    "https://cdn.shopify.com/s/files/1/0522/2239/4534/files/CT21355-22_1024x1024.webp",
    "https://cdn.shopify.com/s/files/1/0522/2239/4534/files/00907857-C6B0-4D2A-8AEA-688BDE1E67D7_1024x1024.jpg",
    "https://cdn.shopify.com/s/files/1/0522/2239/4534/files/Photoroom_20250116_141832_1024x1024.jpg",
    "https://cdn.shopify.com/s/files/1/0522/2239/4534/files/CT21355-22_1024x1024.webp"
]

def download_image_as_pil(url: str, timeout: int = 10) -> Image.Image:
    try:
        response = requests.get(
            url, stream=True, headers=REQUESTS_HEADERS, timeout=timeout
        )
        
        if response.status_code == 200:
            return Image.open(response.raw).convert("RGB")  # Ensure consistent format
        print(f"Failed to download image from {url}. Status code: {response.status_code}")
        return None
    
    except Exception as e:
        print(f"Error downloading image from {url}: {e}")
        return None

def main():
    # Initialize encoder
    encoder = FashionCLIPEncoder()
    
    n_success, n = 0, 0
    point_ids, images, payloads = [], [], []
    loop = tqdm.tqdm(iterable=IMAGE_URLS, total=len(IMAGE_URLS), desc="Processing images")
    
    for url in loop:
        image = download_image_as_pil(url=url)
        
        if isinstance(image, Image.Image):
            point_id = str(uuid.uuid4())
            
            images.append(image)
            payloads.append({"image_url": url, "point_id": point_id})
            point_ids.append(point_id)
        
        if len(point_ids) > 0 and len(point_ids) % BATCH_SIZE == 0:
            n += len(point_ids)
            
            try:
                embeddings = encoder.encode_images(images)
                for i, embedding in enumerate(embeddings):
                    print(f"Embedding for image {payloads[i]['image_url']}: {embedding[:5]}...")  # Print first 5 values
                print(f"Processed batch of {len(images)} images.")
                n_success += len(embeddings)
            except Exception as e:
                print(f"Batch encoding failed: {e}")
            
            # Clear buffers
            point_ids, images, payloads = [], [], []
        
        # Update progress
        loop.set_postfix(success=n_success, processed=n)
    
    # Process remaining images
    if len(point_ids) > 0:
        n += len(point_ids)
        try:
            embeddings = encoder.encode_images(images)
            for i, embedding in enumerate(embeddings):
                print(f"Embedding for image {payloads[i]['image_url']}: {embedding[:5]}...")
            print(f"Processed final batch of {len(images)} images.")
            n_success += len(embeddings)
        except Exception as e:
            print(f"Final batch encoding failed: {e}")
    
    print(f"Completed embedding process: {n_success}/{n} images successfully embedded.")

if __name__ == "__main__":
    main()
