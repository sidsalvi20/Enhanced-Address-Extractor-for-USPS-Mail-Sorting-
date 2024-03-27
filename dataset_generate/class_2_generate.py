import csv
import json
import os
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageFont 
import numpy as np
from faker import Faker

def load_addresses_from_csv(csv_file):
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


def generate_fake_name():
    fake = Faker()
    return fake.name()


def change_view_angle(image_path):
    img = cv2.imread(image_path)

    height, width = img.shape[:2]
    pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    pts2 = np.float32([[0, 0], [width, height * 0.1], [0, height], [width, height * 0.9]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    transformed_img = cv2.warpPerspective(img, matrix, (img.shape[1], img.shape[0]))
    output_path = 'output_images/class_2/changed_angle_' + image_path.split('/')[-1]
    cv2.imwrite(output_path, transformed_img)
    print(f"Image saved: {output_path}")


def create_addressed_envelope(envelope_dir, from_address, to_address, output_dir, json_output_dir, index):
    envelope_images = [f for f in os.listdir(envelope_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    if not envelope_images:
        print("No envelope images found. Please check your envelope_images directory.")
        return
    elif len(envelope_images) != 1:
        print("Please ensure there's only one envelope image in the directory.")
        return
    envelope_image = envelope_images[0]
    image_path = os.path.join(envelope_dir, envelope_image)
    envelope = Image.open(image_path)
    draw = ImageDraw.Draw(envelope)
    font_path = "font_style/Avantgarde-Bold.otf"
    from_font = ImageFont.truetype(font_path, size=10)
    to_font = ImageFont.truetype(font_path, size=10)
    from_position = (70, 150) 
    to_position = (envelope.width // 2 - 20, envelope.height // 2 - 20) 

    from_name = generate_fake_name()
    to_name = generate_fake_name()


    from_address_inp = f"{from_name}\n{from_address['street']}\n{from_address['city']}, {from_address['state']}, {from_address['zip']}"
    to_address_inp = f"{to_name}\n{to_address['street']}\n{to_address['city']}, {to_address['state']}, {to_address['zip']}"
    draw.text(from_position, from_address_inp, fill="black", font=from_font)
    draw.text(to_position, to_address_inp, fill="black", font=to_font)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_image_name = f"addressed_{index}_{os.path.splitext(envelope_image)[0]}.png"
    output_path = os.path.join(output_dir, output_image_name)
    envelope.save(output_path)
    print(f"Envelope created: {output_path}")

    change_view_angle(output_path)

    json_data = {
        "from_address": f"{from_name}, {from_address['street']}, {from_address['city']}, {from_address['state']}, {from_address['zip']}",
        "to_address": f"{to_name}, {to_address['street']}, {to_address['city']}, {to_address['state']}, {to_address['zip']}"
    }
    json_output_name = f"addressed_{index}_{os.path.splitext(envelope_image)[0]}.json"
    json_output_path = os.path.join(json_output_dir, json_output_name)
    with open(json_output_path, 'w') as json_file:
        json.dump(json_data, json_file)

    json_output_path_changed = os.path.join(json_output_dir, 'changed_angle_' + json_output_name)
    with open(json_output_path_changed, 'w') as json_file:
        json.dump(json_data, json_file)

    print(f"JSON data saved: {json_output_path}")



if __name__ == '__main__':

    from_csv_file = 'address_csv.csv'
    addresses = load_addresses_from_csv(from_csv_file)

    envelope_dir = 'input_images/class_2'
    output_dir = 'output_images/class_2'
    json_output_dir = 'output_images/class_2_json'

    for i, from_address in enumerate(addresses):
     
        to_address = random.choice([address for address in addresses if address != from_address])
 
        create_addressed_envelope(envelope_dir, from_address, to_address, output_dir, json_output_dir, i)