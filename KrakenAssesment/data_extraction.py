import fitz  # PyMuPDF
import os
import re
import pandas as pd

class DataExtraction:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path  # Path to the single PDF

    def extract_text_images_from_pdf(self):
        # Step 1: Extract text using PyMuPDF
        document = fitz.open(self.pdf_path)
        text_content = ""

        for page_num in range(len(document)):
            page = document.load_page(page_num)
            text_content += page.get_text()

        pattern = r'(.*?)\n-{100,}'  # Matches any text followed by a line of at least 100 dashes
        # Find all descriptions that match the pattern
        descriptions = re.findall(pattern, text_content, re.DOTALL)
        # Clean up descriptions by stripping excess whitespace
        descriptions = [desc.strip() for desc in descriptions if desc.strip()]
        # Create a dictionary with numeric keys
        text = {i: description for i, description in enumerate(descriptions)}

        # Step 2: Extract individual images from each page using PyMuPDF
        images = []
        output_folder = "extracted_images"
        os.makedirs(output_folder, exist_ok=True)
        for page_num in range(len(document)):
            page = document[page_num]
            image_list = page.get_images(full=True)

            # Extract each image on the page
            for img_index, img in enumerate(image_list):
                xref = img[0]  # Xref for the image
                base_image = document.extract_image(xref)
                image_bytes = base_image["image"]

                # Save the image to the output folder
                image_filename = os.path.join(output_folder, f"page_{page_num}_img_{img_index + 1}.png")
                with open(image_filename, "wb") as image_file:
                    image_file.write(image_bytes)

                images.append(image_filename)

        return text, images

    def clean_text(self, text):
        # Clean the extracted text
        for key, value in text.items():
            cleaned_text = re.sub(r'[^\w\s,.€]', '', value)  # Remove unwanted characters but keep €, commas, etc.
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()  # Remove extra spaces
            text[key] = cleaned_text
        return text

    def parse_property_details(self, property_dict):
        properties = []

        for _, listing in property_dict.items():  # Loop through each dictionary entry

            property_data = {}

            # Extract relevant information using regex with case-insensitivity and flexible formatting
            address_match = re.search(r'in\s*(.*?),\s*ID', listing, re.IGNORECASE)
            price_match = re.search(r'pret\s*[:\-]?\s*([\d,]+(?:\s?€)?)', listing,
                                    re.IGNORECASE)  # Account for uppercase/lowercase and variations
            rooms_match = re.search(r'nr\.\s*camere\s*[:\-]?\s*(\d+)', listing,
                                    re.IGNORECASE)  # Account for case and punctuation variations
            area_match = re.search(r'suprafata\s*utila\s*[:\-]?\s*([\d ]+m2)', listing,
                                   re.IGNORECASE)  # Flexible with case and punctuation
            bathrooms_match = re.search(r'nr\.\s*bai\s*[:\-]?\s*(\d+)', listing, re.IGNORECASE)
            kitchens_match = re.search(r'nr\.\s*bucatarii\s*[:\-]?\s*(\d+)', listing, re.IGNORECASE)

            # Populate the property_data dictionary
            property_data['address'] = address_match.group(1) if address_match else None
            property_data['price'] = price_match.group(1) if price_match else None
            property_data['rooms'] = int(rooms_match.group(1)) if rooms_match else None
            property_data['area'] = area_match.group(1) if area_match else None
            property_data['bathrooms'] = int(bathrooms_match.group(1)) if bathrooms_match else None
            property_data['kitchens'] = int(kitchens_match.group(1)) if kitchens_match else None

            properties.append(property_data)
        return properties

    def extract_to_dataframe(self):
        # Extract text and images, and structure the data in a DataFrame
        text_content, images = self.extract_text_images_from_pdf()
        cleaned_text = self.clean_text(text_content)

        # Parse properties
        parsed_properties = self.parse_property_details(cleaned_text)

        # Create a DataFrame with the parsed property data
        properties_df = pd.DataFrame(parsed_properties)

        # Add images column: Associate each property with its corresponding images
        grouped_images = {}
        index = 0
        for image_filename in images:
            # Extract the page number from the image filename (e.g., 'page_1_img_1.png')
            page_num = int(re.search(r'page_(\d+)_', image_filename).group(1))
            # Add the image to the corresponding page's list of images
            if page_num not in grouped_images:
                grouped_images[page_num] = []
            grouped_images[page_num].append(image_filename)

        # Add images column: Associate each property with its corresponding images
        property_images = []

        # Loop through the properties and assign images dynamically based on page number

        for key in grouped_images:
            property_images.append(grouped_images[key])
        # for index, row in properties_df.iterrows():
        #
        #     # If images exist for this page, assign them to the property
        #     images_for_property = grouped_images.get(index+1, [])
        #
        #     property_images.append(images_for_property)

        # Add images to the DataFrame
        properties_df['images'] = property_images

        return properties_df

