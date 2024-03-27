import csv
import json
import os
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageFont 
import numpy as np
from faker import Faker


def load_addresses_from_csv_class_3(csv_file):
    addresses = []
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            address = {
                'street': row['address'],
                'city': row['city'],
                'state': row['state'],
                'zip': row['zip']
            }
            addresses.append(address)
    return addresses


def generate_fake_name_class_3():
    fake = Faker()    
    return fake.name()


def apply_camera_effect_class_3(img):
 
    width, height = img.size

    vignette = Image.new("RGB", (width, height), color='black') 
    draw = ImageDraw.Draw(vignette)
    
    radius = max(width, height) / 2
    for i in range(width):
        for j in range(height):
            distance = np.sqrt((i - width / 2) ** 2 + (j - height / 2) ** 2)
            distance = distance / radius
            opacity = 255 - int(distance * 255)
            draw.point((i, j), fill=(opacity, opacity, opacity))
    
    vignette = vignette.convert("L")
    img.putalpha(vignette) 
    img = img.convert("RGB") 

    noise = np.random.normal(0, 25, (height, width, 3))
    np_img = np.array(img, dtype=np.float32)
    np_img += noise
    np_img = np.clip(np_img, 0, 255).astype(np.uint8)
    img = Image.fromarray(np_img)

    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(0.9)  
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.1) 

    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    return img


def change_view_angle_class_3(image_path):

    img = cv2.imread(image_path)

    height, width = img.shape[:2]
    pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    pts2 = np.float32([[0, 0], [width, height * 0.1], [0, height], [width, height * 0.9]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    transformed_img = cv2.warpPerspective(img, matrix, (img.shape[1], img.shape[0]))
    output_path = 'output_images/class_3/changed_angle_' + image_path.split('/')[-1]
    cv2.imwrite(output_path, transformed_img)
    print(f"Image saved: {output_path}")




def create_addressed_envelope_class_3(envelope_dir, from_address, to_address, output_dir, json_output_dir, index, font_list):
    envelope_images = [f for f in os.listdir(envelope_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    if not envelope_images:
        print("No envelope images found. Please check your envelope_images directory.")
        return
    elif len(envelope_images) != 1:
        print("Please ensure there's only one envelope image in the directory.")
        return

    envelope_image = envelope_images[0]
    image_path = os.path.join(envelope_dir, envelope_image)

    from_name = generate_fake_name_class_3()
    to_name = generate_fake_name_class_3()
    envelope1 = Image.open(image_path)
        
    positions = [
        ((40, 120), (envelope1.width // 2 - 100, envelope1.height // 2 - 30)),
        ((150, 120), (envelope1.width // 2 + 60, envelope1.height // 2 - 30))
    ]



    for j, font_path in enumerate(font_list):
        for k, (from_position, to_position) in enumerate(positions):
            # Save non-rotated image
            save_addressed_envelope(image_path, from_name, to_name, from_address, to_address, font_path, from_position, to_position, output_dir, json_output_dir, index, j, k, rotation=0)
            
            # Additionally save a rotated version in some iterations
            if random.choice([True, False]):  # Randomly decide to rotate
                rotation_angle = random.uniform(-40, 40)  # Random angle between -40 and 40 degrees
                save_addressed_envelope(image_path, from_name, to_name, from_address, to_address, font_path, from_position, to_position, output_dir, json_output_dir, index, j, k, rotation=rotation_angle)


def save_addressed_envelope(image_path, from_name, to_name, from_address, to_address, font_path, from_position, to_position, output_dir, json_output_dir, index, font_index, position_index, rotation):
    
    envelope = Image.open(image_path)
    draw = ImageDraw.Draw(envelope)
    from_font = ImageFont.truetype(font_path, size=12)
    to_font = ImageFont.truetype(font_path, size=12)

    from_address_text = f"From :\n{from_name}\n{from_address['street']}\n{from_address['city']}, {from_address['state']}, {from_address['zip']}"
    to_address_text = f"To :\n{to_name}\n{to_address['street']}\n{to_address['city']}, {to_address['state']}, {to_address['zip']}"

    draw.text(from_position, from_address_text, fill="black", font=from_font)
    draw.text(to_position, to_address_text, fill="black", font=to_font)

    envelope = apply_camera_effect_class_3(envelope)

    if rotation != 0:
        envelope = envelope.rotate(rotation, expand=True, fillcolor='white')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    rotation_suffix = "_rotated" if rotation != 0 else ""
    output_image_name = f"addressed_{index}_font_{font_index}_pos_{position_index}{rotation_suffix}.png"
    output_path = os.path.join(output_dir, output_image_name)
    envelope.save(output_path)


    # Save JSON file with the same name as the image
    json_data = {
        "from_address": f"{from_name}, {from_address['street']}, {from_address['city']}, {from_address['state']}, {from_address['zip']}",
        "to_address": f"{to_name}, {to_address['street']}, {to_address['city']}, {to_address['state']}, {to_address['zip']}"
    }

    if not os.path.exists(json_output_dir):
        os.makedirs(json_output_dir)
    
    json_output_name = f"addressed_{index}_font_{font_index}_pos_{position_index}{rotation_suffix}.json"
    json_output_path = os.path.join(json_output_dir, json_output_name)
    
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data, json_file)
    
    if rotation == 0:
        change_view_angle_class_3(output_path)
        json_output_path_changed = os.path.join(json_output_dir, 'changed_angle_' + json_output_name)
        with open(json_output_path_changed, 'w') as json_file:
            json.dump(json_data, json_file)

    
    
    print(f"Envelope image saved: {output_path}")
    print(f"JSON data saved: {json_output_path}")


if __name__ == '__main__':

    font1 = "font_style/Avantgarde-Bold.otf"
    font2 = "font_style/DejaVuSans-Bold.ttf"
    font3 = "font_style/GOODDP__.TTF"
    font4 = "font_style/GreatVibes-Regular.ttf"
    font5 = "font_style/NexaRustHandmade-Extended.otf"
    font6 = "font_style/PrincessSofia-Regular.ttf"

    font_list = [font1, font2, font3, font4, font5, font6]
    from_csv_file = 'address_csv.csv'
    addresses = load_addresses_from_csv_class_3(from_csv_file)

    envelope_dir = 'input_images/class_3'
    output_dir = 'output_images/class_3'
    json_output_dir = 'output_images/class_3_json'

    for i, from_address in enumerate(addresses):
      
        to_address = random.choice([address for address in addresses if address != from_address])

        create_addressed_envelope_class_3(envelope_dir, from_address, to_address, output_dir, json_output_dir, i, font_list)

