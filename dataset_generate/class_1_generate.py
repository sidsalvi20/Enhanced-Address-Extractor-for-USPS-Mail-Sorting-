import csv
import json
import os
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageFont 
import numpy as np
from faker import Faker


def load_addresses_from_csv_class_1(csv_file):
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


def generate_fake_name_class_1():
    fake = Faker()
    return fake.name()


def change_view_angle_class_1(image_path):
    
    img = cv2.imread(image_path)

    height, width = img.shape[:2]
    pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    pts2 = np.float32([[0, 0], [width, height * 0.1], [0, height], [width, height * 0.9]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    transformed_img = cv2.warpPerspective(img, matrix, (img.shape[1], img.shape[0]))

    output_path = 'output_images/class_1/changed_angle_' + image_path.split('/')[-1]
    cv2.imwrite(output_path, transformed_img)
    print(f"Image saved: {output_path}")


def create_addressed_envelope_class_1(envelope_dir, from_address, to_address, output_dir, json_output_dir, index):

    envelope_images = [f for f in os.listdir(envelope_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    if not envelope_images:
        print("No envelope images found. Please check your envelope_images directory.")
        return
    elif len(envelope_images) != 1:
        print("Please ensure there's only one envelope image in the directory.")
        return


    from_name = generate_fake_name_class_1()
    to_name = generate_fake_name_class_1()
    j = 0
    for font_path in font_list:
      envelope_image = envelope_images[0]
      image_path = os.path.join(envelope_dir, envelope_image)

      envelope = Image.open(image_path)
      draw = ImageDraw.Draw(envelope)

      from_font = ImageFont.truetype(font_path, size=11)
      to_font = ImageFont.truetype(font_path, size=11)

      from_position = (envelope.width - 180, 120)
      to_position = (envelope.width - 180, 250) 

      from_address_inp = f"{from_name}\n{from_address['street']}\n{from_address['city']}, {from_address['state']}, {from_address['zip']}"
      to_address_inp = f"{to_name}\n{to_address['street']}\n{to_address['city']}, {to_address['state']}, {to_address['zip']}"

      draw.text(from_position, from_address_inp, fill="black", font=from_font)
      draw.text(to_position, to_address_inp, fill="black", font=to_font)

      if not os.path.exists(output_dir):
          os.makedirs(output_dir)
      output_image_name = f"addressed_{index}_font_{j}_{os.path.splitext(envelope_image)[0]}.png"
      output_path = os.path.join(output_dir, output_image_name)
      envelope.save(output_path)
      print(f"Envelope created: {output_path}")

      change_view_angle_class_1(output_path)

      json_data = {
        "from_address": f"{from_name}, {from_address['street']}, {from_address['city']}, {from_address['state']}, {from_address['zip']}",
        "to_address": f"{to_name}, {to_address['street']}, {to_address['city']}, {to_address['state']}, {to_address['zip']}"
      }
      json_output_name = f"addressed_{index}_font_{j}_{os.path.splitext(envelope_image)[0]}.json"
      json_output_path = os.path.join(json_output_dir, json_output_name)
      with open(json_output_path, 'w') as json_file:
          json.dump(json_data, json_file)

      json_output_path_changed = os.path.join(json_output_dir, 'changed_angle_' + json_output_name)
      with open(json_output_path_changed, 'w') as json_file:
          json.dump(json_data, json_file)

      print(f"JSON data saved: {json_output_path}")
      j = j + 1


if __name__ == '__main__':

    font1 = "font_style/Avantgarde-Bold.otf"
    font2 = "font_style/DejaVuSans-Bold.ttf"
    font3 = "font_style/GOODDP__.TTF"
    font4 = "font_style/GreatVibes-Regular.ttf"
    font5 = "font_style/NexaRustHandmade-Extended.otf"
    font6 = "font_style/PrincessSofia-Regular.ttf"

    font_list = [font1, font2, font3, font4, font5, font6]
    from_csv_file = 'address_csv.csv'
  
    envelope_dir = 'input_images/class_1'
    output_dir = 'output_images/class_1'
    json_output_dir = 'output_images/class_1_json'
    
    addresses = load_addresses_from_csv_class_1(from_csv_file)


    for i, from_address in enumerate(addresses):
       
        to_address = random.choice([address for address in addresses if address != from_address])
       
        create_addressed_envelope_class_1(envelope_dir, from_address, to_address, output_dir, json_output_dir, i)