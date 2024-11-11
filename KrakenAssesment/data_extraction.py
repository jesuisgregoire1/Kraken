import fitz
import os
import re
import pandas as pd


class DataExtraction:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def extract_text_images_from_pdf(self):
        document = fitz.open(self.pdf_path)
        text_content = ""

        for page_num in range(len(document)):
            page = document.load_page(page_num)
            text_content += page.get_text()

        pattern = r'(.*?)\n-{100,}'  # matches any text followed by a line of at least 100 dashes
        descriptions = re.findall(pattern, text_content, re.DOTALL)
        descriptions = [desc.strip() for desc in descriptions if desc.strip()]
        text = {i: description for i, description in enumerate(descriptions)}

        images = []
        output_folder = "extracted_images"
        os.makedirs(output_folder, exist_ok=True)
        for page_num in range(len(document)):
            page = document[page_num]
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = document.extract_image(xref)
                image_bytes = base_image["image"]
                image_filename = os.path.join(output_folder, f"page_{page_num}_img_{img_index + 1}.png")
                with open(image_filename, "wb") as image_file:
                    image_file.write(image_bytes)
                images.append(image_filename)
        return text, images

    def clean_text(self, text):
        # Clean the extracted text
        for key, value in text.items():
            cleaned_text = re.sub(r'[^\w\s,.€]', '', value)
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
            text[key] = cleaned_text
        return text

    def parse_property_details(self, property_dict):
        properties = []
        for _, listing in property_dict.items():

            property_data = {}
            # extract relevant information
            address_match = re.search(r'in\s*(.*?),\s*ID', listing, re.IGNORECASE)
            price_match = re.search(r'pret\s*[:\-]?\s*([\d,]+(?:\s?€)?)', listing, re.IGNORECASE)
            rooms_match = re.search(r'nr\.\s*camere\s*[:\-]?\s*(\d+)', listing, re.IGNORECASE)
            area_match = re.search(r'suprafata\s*utila\s*[:\-]?\s*([\d ]+m2)', listing, re.IGNORECASE)
            bathrooms_match = re.search(r'nr\.\s*bai\s*[:\-]?\s*(\d+)', listing, re.IGNORECASE)
            kitchens_match = re.search(r'nr\.\s*bucatarii\s*[:\-]?\s*(\d+)', listing, re.IGNORECASE)

            # populate the property_data dictionary
            property_data['address'] = address_match.group(1) if address_match else None
            property_data['price'] = price_match.group(1) if price_match else None
            property_data['rooms'] = int(rooms_match.group(1)) if rooms_match else None
            property_data['area'] = area_match.group(1) if area_match else None
            property_data['bathrooms'] = int(bathrooms_match.group(1)) if bathrooms_match else None
            property_data['kitchens'] = int(kitchens_match.group(1)) if kitchens_match else None

            properties.append(property_data)
        return properties

    def extract_to_dataframe(self):
        # extract text and images into a DataFrame
        text_content, images = self.extract_text_images_from_pdf()
        cleaned_text = self.clean_text(text_content)
        # parse properties
        parsed_properties = self.parse_property_details(cleaned_text)
        # create a DataFrame with the parsed property data
        properties_df = pd.DataFrame(parsed_properties)
        # add images column: Associate each property with its corresponding images
        grouped_images = {}
        for image_filename in images:
            page_num = int(re.search(r'page_(\d+)_', image_filename).group(1))
            if page_num not in grouped_images:
                grouped_images[page_num] = []
            grouped_images[page_num].append(image_filename)

        property_images = []
        for key in grouped_images:
            property_images.append(grouped_images[key])

        # add images to the DataFrame
        properties_df['images'] = property_images

        return properties_df
