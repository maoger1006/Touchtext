import mediapipe as mp
import cv2
import time
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as functional
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision.models import vgg16
from torchvision.datasets import FashionMNIST
from torch import optim
from os.path import dirname, join as pjoin
from scipy.io import loadmat
import os
from PIL import Image, ImageDraw, ImageFont
import math
import matplotlib.pyplot as plt
from matplotlib import cm
import random
import math
import re
from transformers import pipeline
from textblob import TextBlob

# from google.colab.patches import cv2_imshow




class handTracker():
    def __init__(self, mode=False, maxHands=1, detectionCon=0.5,modelComplexity=1,trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def handsFinder(self, image, draw=False):
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:

                if draw:
                    self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)
        return image

    def positionFinder(self, image, handNo=0, draw=True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            Hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(Hand.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
            # if draw:
            #     cv2.circle(image, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        return lmlist


def get_countdown_text(start_time, countdown_seconds=10):
    """ Returns the countdown text based on the current time and start time. """
    elapsed_time = time.time() - start_time
    remaining_time = max(countdown_seconds - int(elapsed_time), 0)
    return str(remaining_time) if remaining_time >= 0 else ""

class ImageDataset(Dataset):
    def __init__(self, root_file, labels, transforms):
        self.image_file = root_file
        self.label = labels
        self.transform = transforms

        self.image_files_all = []
        total_folders = sorted(os.listdir(root_file))
        last_folders = total_folders[-26:]
        # for class_idx, class_dir in enumerate(sorted(os.listdir(root_file))):
        for class_idx, class_dir in enumerate(last_folders):
            classfull = os.path.join(root_file, class_dir)
            if os.path.isdir(classfull):
                for f in sorted(os.listdir(classfull)):
                    f2 = os.path.join(classfull, f)
                    self.image_files_all.append(f2)

    def __len__(self):
        return len(self.image_files_all)

    def __getitem__(self, idx):
        image_path = self.image_files_all[idx]
        label = self.label[idx]
        image = Image.open(image_path).convert("RGB")
        # image2 = self.img_transform(image)
        return image, label

    def img_transform(self, image):
        img = self.transform(image)
        return img

def get_image(pic_in):
    # Get image
    # pic = Image.open(path).convert("RGB")
    pic = np.array(pic_in)


    # RGB to gray and threholding
    gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Opening the gaps between letter
    # kernel = np.ones((3,3),np.uint8)
    # thresh = cv2.erode(thresh,kernel,iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Get the contours coordinates
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Sort the countour
    sorted_ctrs = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0] - cv2.boundingRect(ctr)[1])
    # Get the letter from image
    pic_all = []
    cod = []
    for c in sorted_ctrs:
        x, y, w, h = cv2.boundingRect(c)
        cod.append((x, y, w, h))

    # Get sorted coordinate
    sort_result, spaces, locs, text_space_lines = sort_findspace(cod)
    new_group = flatten_list(text_space_lines)

    # Get the image for each letter
    pic_all = get_image2(thresh, pic, new_group)

    return thresh, pic_all, cod

def flatten_list(group):
  new_group = []
  for i in range(len(group)): #3
    for j in range(len(group[i])):
      new_group.append(group[i][j])

  return new_group


def get_image2(thresh,image,group):
  pic_all = []
  for i in group:
    #draw the bounding box
    cv2.rectangle(thresh,(i[0]-2,i[1]-2),(i[0]+i[2]+2,i[1]+i[3]+2),(100,100,100),2)

    #crop the letter image and append them to a list
    letter_image = image[i[1]:i[1]+i[3],i[0]:i[0]+i[2]]
    pic_all.append(letter_image)
    print(letter_image.shape)

  return pic_all

def distance(coordinates):
  carray = np.array(coordinates)
  d = np.linalg.norm(carray[0:2])

  return d


def paras(coordinates):
  carray = np.array(coordinates)
  topleft = [carray[0],carray[1]]
  topright = [carray[0]+carray[2],carray[1]]
  bottomleft = [carray[0],carray[1]+carray[3]]
  bottomright = [carray[0]+carray[2],carray[1]+carray[3]]
  center = [carray[0]+carray[2]/2,carray[1]+carray[3]/2]
  return center,topleft,topright,bottomleft,bottomright

######################################################################################
def sort_findspace(l):
  distances = np.zeros([len(l),1])
  for i in range(len(l)):
    distances[i,:] = distance(l[i])


  indices = np.argsort(distances,axis = 0)


  #pick the smallest distance and add it to the first line
  lines = []
  first_line = []
  first_line.append(indices[0][0])


  #calculate all parameters for every character(contour)
  parameters = []
  for i in range(len(l)):
    parameters.append(paras(l[i]))


  #determine the first line
  not_this_line = []
  for i in range(len(l)):
    if i == first_line[0]:
      continue
    else:
      if (list(parameters[i])[0][1] <= list(parameters[first_line[0]])[4][1]) and (list(parameters[i])[0][1] >= list(parameters[first_line[0]])[1][1]):
        first_line.append(i)
      else:
        not_this_line.append(i)

  lines.append(first_line)


  #determine the rest lines
  #first, we should check if we have more than 1 line

  if len(not_this_line) == 0:
    pass
  else:
    for k in range(10):
      new_line = []
      not_this_new_line = []
      new_line.append(not_this_line[0])
      for i in range(len(not_this_line)-1):
        j = not_this_line[i+1]
        if (list(parameters[j])[0][1] <= list(parameters[new_line[0]])[4][1]) and (list(parameters[j])[0][1] >= list(parameters[new_line[0]])[1][1]):
          new_line.append(j)
        else:
          not_this_new_line.append(j)
      not_this_line = not_this_new_line

      lines.append(new_line)

      if len(not_this_line) == 0:
        break
      else:
        continue


  #sort between lines
  row_num = len(lines)
  first_letters_y = np.zeros([row_num,1])
  for i in range(first_letters_y.shape[0]):
    first_letters_y[i] = list(parameters[lines[i][0]])[0][1]

  sorted_first_letters_y_indices = np.argsort(first_letters_y,axis = 0)
  final_lines = []
  for i in range(row_num):
    final_lines.append(lines[sorted_first_letters_y_indices[i][0]])



  #now sort inside specific line
  sorted_final_lines = []
  for i in range(row_num):
    temp = np.zeros([len(final_lines[i]),1])
    for j in range(len(final_lines[i])):
      temp[j,:] = list(parameters[final_lines[i][j]])[0][0]
    #print(np.argsort(temp,axis = 0))

    row = np.array(final_lines[i])[np.argsort(temp,axis = 0)]
    sorted_final_lines.append(list(row))

  #make result more readable
  result = []
  for i in range(row_num):
    rowi = []
    for j in range(len(sorted_final_lines[i])):
      rowi.append(sorted_final_lines[i][j][0])
    result.append(rowi)


  #now claculate the distance between each characters:
  cds = []
  for i in range(row_num):
    cd = []
    for j in range(len(result[i])-1):
      cd.append(list(parameters[result[i][j+1]])[3][0]-list(parameters[result[i][j]])[4][0])

    cds.append(cd)

    sort_result = result
    spaces = cds

  all = []
  for i in range(len(spaces)):
    for j in range(len(spaces[i])):
      all.append(spaces[i][j])

  all_array = np.array(all)

  #now output the location of space
  space_locs = []
  for i in range(len(spaces)):
    space_loc = []
    for j in range(len(spaces[i])):
      if spaces[i][j] < 2*np.mean(all_array):
        pass
      else:
        space_loc.append(j+1)

    space_locs.append(space_loc)

  locs = space_locs


#calculate the space coordinates
  space_coordinates = []
  for i in range(len(locs)):
    space_coordinates_each_line = []
    if len(locs[i]) == 0:
      space_coordinates.append(space_coordinates_each_line)
    else:
      for j in range(len(locs[i])):
        space_coordinates_each_line.append((list(parameters[sort_result[i][locs[i][j]-1]])[2][0],list(parameters[sort_result[i][locs[i][j]-1]])[2][1],spaces[i][locs[i][j]-1],list(l[sort_result[i][locs[i][j]-1]])[3]))
      space_coordinates.append(space_coordinates_each_line)

#insert space into the text
  text_space_lines = []
  for i in range(len(sort_result)):
    text_space_line = []
    for j in range(len(sort_result[i])):
      text_space_line.append(l[sort_result[i][j]])
    for k in range(len(space_coordinates[i])):
      text_space_line.append(space_coordinates[i][k])
    text_space_lines.append(text_space_line)

#now sort the text and space
  for i in range(len(text_space_lines)):
    text_space_lines[i].sort()

#since we add space in each lines, the index for the space should be modified
  for i in range(len(locs)):
    if len(locs[i]) == 0:
      pass
    else:
      for j in range(len(locs[i])):
        locs[i][j] += j



  return sort_result,spaces,locs,text_space_lines

##################################################################################################
def convert_odd_to_even(number):
  if number % 2 != 0:
   return number-1
  else:
    return number


def convert(image):
    # Check and adjust the height if it's odd
    if image.shape[0] % 2 != 0:
        image = image[0:image.shape[0]-1, :]

    # Check and adjust the width if it's odd
    if image.shape[1] % 2 != 0:
        image = image[:, 0:image.shape[1]-1]

    return image

def enlarge_pic(pic_all,index):
  # print(f'picall{pic_all[2].shape}')
  gray2 = cv2.cvtColor(pic_all[index], cv2.COLOR_BGR2GRAY)
  gray2 = convert(gray2)
  # Create a new blank image with the same dimensions as the original
  h = convert_odd_to_even(gray2.shape[0])

  w = convert_odd_to_even(gray2.shape[1])

  unit = 1.3
  if h>=w:
    new_h = math.ceil(unit*(h))
  elif h<w:
    new_h = math.ceil(unit*(w))

  new_image = np.ones((new_h,new_h))
  new_image[new_image ==1] =255

  #Find the middle point of the original and new
  y_mean = int(h/2)
  x_mean = int(w/2)

  new_mean = int(new_h/2)

  # Place the cropped 'T' in the center of the new blank image
  new_image[new_mean-y_mean:new_mean+y_mean,new_mean-x_mean:new_mean+x_mean] = gray2
  image_t = np.repeat(new_image[:,:,np.newaxis],3,axis=2)


  return image_t



def get_labels():
  #Create labels

  # #create A-Z and a-z
  # labels_A = []
  # for char in range(ord("A"),ord("Z")+1):
  #   labels_A.extend(chr(char)*1016)

  labels_a = []
  for char in range(ord("a"),ord("z")+1):
    labels_a.extend(chr(char)*1016)

  #Extend and get the labels
  # labels_num.extend(labels_A)


  return labels_a

def mapping_pred(labels,pred):
  #define mapping between num and string
  lb = np.unique(labels)
  mapping = {label:index for index, label in enumerate(lb)}
  reverse_mapping = {index:label for label, index in mapping.items()}

  #Convert pred to string
  pred_all = []
  for i in pred:
    if i == 100:
      pred_all.extend(" ")
    else:
      pred_string = reverse_mapping[i]
      pred_all.extend(pred_string)
  sentence = "".join(str(i) for i in pred_all)

  return sentence



def refill (image):
  D = image.shape[1]
  pixel = image[0,0,:]
  image_refilled = np.zeros([D,D,3])
  image_refilled[:,:,:] = pixel
  aaa = int(((image.shape[1]-image.shape[0])/2))
  bbb = int(((image.shape[1]+image.shape[0])/2))
  image_refilled[aaa:bbb,:] = image
  return image_refilled



def put_text_with_pil(frame, text, position, font_path, font_size, font_color):
    # Convert the OpenCV image to PIL format
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)

    # Specify the font (use a path to a .ttf file that supports French characters)
    font = ImageFont.truetype(font_path, font_size)

    # Draw the text
    draw.text(position, text, font=font, fill=font_color)

    # Convert back to OpenCV format
    return np.array(img_pil)


def main():
    cap = cv2.VideoCapture(1)
    tracker = handTracker()
    first_point = None
    second_point = None
    extend_x = 0
    extend_y = 0
    start_time = time.time()
    counter = 0
    display_string = ''
    cropped_image = None
    counter1 = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    def word_prediction(pic_all, model, img_transform, locs):
        word = []
        for i in range(len(pic_all)):
            if i in locs[0]:
                word.append(100)
            else:
                # Convert np_array to PIL image and tranform them before putting into model
                pic = enlarge_pic(pic_all, i)
                PIL_image = Image.fromarray(np.uint8(pic)).convert('RGB')
                PILtf = img_transform(PIL_image)
                PILtf = PILtf.unsqueeze(0)

                # Find the prediction label from the model
                PILtf = PILtf.to(device)
                pred = model(PILtf)

                # Find the correct prediction type from the Softmax output
                _, indices = torch.max(pred, axis=1)
                index = indices.cpu().item()
                word.append(index)

        return word

    def word_prediction_without(pic_all, model, img_transform):
        word = []
        for i in range(len(pic_all)):
            # Convert np_array to PIL image and tranform them before putting into model
            pic = enlarge_pic(pic_all, i)
            PIL_image = Image.fromarray(np.uint8(pic)).convert('RGB')
            PILtf = img_transform(PIL_image)
            PILtf = PILtf.unsqueeze(0)

            # Find the prediction label from the model
            PILtf = PILtf.to(device)
            pred = model(PILtf)

            # Find the correct prediction type from the Softmax output
            _, indices = torch.max(pred, axis=1)
            index = indices.cpu().item()
            word.append(index)
        return word

    vgg26 = torch.load("C:\\Users\\22770\\Desktop\\MLSP project\\vgg26.pth")

    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    while True:
        success, image = cap.read()
        if not success:
            break

        image = tracker.handsFinder(image)
        lmList = tracker.positionFinder(image)
        if (len(lmList) != 0):

            extend_x = lmList[8][1] * 2 - lmList[7][1]
            extend_y = lmList[8][2] * 2 - lmList[7][2]

        cv2.circle(image, (extend_x, extend_y), 10, (255, 0, 255), cv2.FILLED)

        current_time = time.time()
        countdown_text = get_countdown_text(start_time)
        if (current_time - start_time) <= 10:
            cv2.putText(image, 'time remains: ' + countdown_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        if (current_time - start_time) >= 10 and first_point is None:
            first_point = [extend_x, extend_y]
            print(first_point)
            midtime = time.time()


        if first_point is not None and second_point is None:
            cv2.circle(image, (first_point[0], first_point[1]), 10, (255, 0, 255), cv2.FILLED)
            cv2.rectangle(image, (first_point[0], first_point[1]), (extend_x, extend_y), (255, 255, 255), 1)

            countdown_text1 = get_countdown_text(midtime)
            cv2.putText(image, 'time remains: ' + countdown_text1, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            if (current_time - midtime) >= 10 and second_point is None:
                second_point = [extend_x, extend_y]
                print(second_point)

        if second_point is not None:

            cv2.rectangle(image, (first_point[0], first_point[1]), (second_point[0], second_point[1]), (255, 255, 255), 1)

            if counter == 0:
                cropped_image = image[first_point[1] + 10:second_point[1]-10, first_point[0]+10:second_point[0] - 10]
                print(cropped_image.shape)
                cv2.imwrite('C:\\Users\\22770\\Desktop\\MLSP project\\output_image.jpg', cropped_image)
                picture_in = Image.open('C:\\Users\\22770\\Desktop\\MLSP project\\output_image.jpg').convert("RGB")
                counter += 1

        if counter1 == 0:
            if cropped_image is not None:

                print(f"string{cropped_image.shape}")
                # picture_in = Image.open('C:\\Users\\22770\\Desktop\\MLSP project\\output_image.jpg').convert("RGB")
                picture_in = np.array(picture_in)
                roi = refill(picture_in)

                resize = roi
                print(resize.shape)
                # Set color thresholds for each channel
                lower_threshold = np.array([150, 150, 150])
                upper_threshold = np.array([255, 255, 255])

                # Create a binary mask based on color thresholds
                color_mask = cv2.inRange(resize, lower_threshold, upper_threshold)

                # Find contours and filter
                contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Filter contours based on height
                HTHRESH = int(roi.shape[0] * 0.1)
                digits = [contour for contour in contours if cv2.boundingRect(contour)[3] > HTHRESH]


                # Take the convex hull of the digit contours
                if digits:
                    hull = cv2.convexHull(np.concatenate(digits))
                else:
                    hull = np.array([])

                # Prepare a mask

                digitsRegion = [hull]
                digitsMask = np.zeros_like(color_mask)
                cv2.drawContours(digitsMask, digitsRegion, 0, 255, -1)

                # Expand the mask to include any information lost in earlier morphological opening
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                digitsMask = cv2.dilate(digitsMask, kernel)

                # Copy the region to get a cleaned image
                cleaned = np.zeros_like(color_mask)
                cleaned = cv2.bitwise_and(color_mask, color_mask, mask=digitsMask)
                cleaned = np.repeat(cleaned[:,:,np.newaxis],3,axis=2)

                cv2.imwrite('C:\\Users\\22770\\Desktop\\MLSP project\\cleaned.jpg', cleaned)
                translator = pipeline("translation_en_to_fr")

                # English text to be translated
                # english_text = "Hello, how are you?"
                #
                # # Perform translation
                # translated_text = translator(english_text)

                # # Print the translated text
                # print(translated_text[0]['translation_text'])

                thresh, pic_all, cod = get_image(cleaned)

                sort_result, spaces, locs, text_space_lines = sort_findspace(cod)
                new_group = flatten_list(text_space_lines)
                word = word_prediction(pic_all, vgg26, img_transform, locs)
                labels = get_labels()
                sentence = mapping_pred(labels, word)
                # print(f"Text Recognition: {sentence}")

                # Correct sentence
                blob = TextBlob(sentence)
                blobc = blob.correct()

                # Convert to string
                blobc = blobc.string
                blobc

                # print(blobc)
                print(f"Text Recognition: {blobc}")
                english_text = blobc

                # Perform translation
                translated_text = translator(english_text)
                print(translated_text)
                # Print the translated text
                # print(f"French Translation: {translated_text[0]['translated_text']}")
                print(translated_text[0])

                # Using .get() to avoid KeyError if 'translated_text' key is not present
                s1 = slice(22, -2)
                display_string = str(translated_text[0])[s1]
                counter1 += 1

                # img1 = np.ones((300, 600))
                img1 = np.full((80, 600), 220)

                img = Image.fromarray(np.uint8(img1)).convert('RGB')
                # Create a drawing object
                draw = ImageDraw.Draw(img)
                fontsize = 60
                font1 = ImageFont.truetype("C:\\Users\\22770\\Desktop\\MLSP project\\fishfryrochesterny_arial\\arial.ttf", fontsize)

                # Define the text and font
                text = display_string

                # To use a specific font, replace the above line with:
                # font = ImageFont.truetype('/path/to/font.ttf', 32)  # Specify font path and size

                # Define text position and color
                position = (30, (img1.shape[0]-fontsize) / 2)  # Change as needed
                color = 'black'  # Change as needed (e.g., (255, 255, 255) for white)

                # Add text to image
                draw.text(position, text, fill=color, font=font1, align="center")

                cv2.imwrite('C:\\Users\\22770\\Desktop\\MLSP project\\draw.jpg',np.array(img))

        # if display_string is not None:
        if counter1 == 1:
            overlay_image = cv2.imread("C:\\Users\\22770\\Desktop\\MLSP project\\draw.jpg")
            # cv2.imwrite('C:\\Users\\22770\\Desktop\\MLSP project\\overlay_image.jpg', overlay_image)
            overlay_image = cv2.resize(overlay_image, (second_point[0] - first_point[0], second_point[1] - first_point[1]))

            rows, cols, channels = overlay_image.shape
            roi = image[first_point[1]:first_point[1] + rows, first_point[0]:first_point[0] + cols]
            image[first_point[1]:first_point[1] + rows, first_point[0]:first_point[0] + cols] = overlay_image

        # cv2.putText(image,  display_string, first_point, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),2)
            # draw = ImageDraw.Draw(image)
            # draw.text(first_point, display_string,  fill=(255, 0, 0))



        cv2.imshow("Video", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
# print(f"Text Recognition: {sentence}")



if __name__ == "__main__":
    main()

