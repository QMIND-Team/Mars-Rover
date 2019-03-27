*A brief description of image generation workflow*

## Scraper

**INPUT: None**

- 20 balls, 100 b/g images (start with 2 balls and 4 images)
- Some b/g images will not have balls embedded, and will just be sent to preprocessing

**OUTPUT: Raw ball and raw background images**

## Ball Generator

**INPUT: Raw ball image**

- Function finds ball in b/g image

**OUTPUT: Image of isolated ball**

## Embeddor

**INPUT: generated ball image, raw background image**

- Each ball will map to each image (1:n)
- Crop image to be square
- Adjust brightness (other features as necessary) to be consistent between ball and background image
- Scale ball image to ~1/32 the size of the raw b/g image
- Embed ball in background image

**OUTPUT: Tennis ball embedded in b/g image (b/g image still original size)**

## Image Preprocessing

**INPUT: Embedded image**

- Preprocess non-ball background images to be cropped and scaled to 128x128 (all that needs to be done for those)
- Scale image down to 128x128

**OUTPUT: Scaled and embedded image (final output)**
