from __future__ import print_function
import os
import cv2
import random
import argparse
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir",  type=str, default='/home/tony/Documents/friends_stuff/ahmad_vclab/i-LIDS-VID/sequences', help="where the dataset is stored")
parser.add_argument("--data_name",    type=str, default='dataset_1', help="Dataset name")
parser.add_argument("--gen_opt_flow", type=str, default='False',  help="Compute optical flow")
parser.add_argument("--train_test_split", type=int, default=0.5,  help="Train/Test split")

args = parser.parse_args()
print('----------------------------------------')
print('FLAGS:')
for arg in vars(args):
    print("'", arg,"'", ": ", getattr(args, arg))
print('----------------------------------------')

def optical_flow_image_gen(optflw, curr_image):
    hsv = np.zeros_like(curr_image)
    hsv[:, :, 2] = 0

    # Separate optFlow into mag and phase components
    mag, ang = cv2.cartToPolar(optflw[:, :, 0], optflw[:, :, 1])
    hsv[:, :, 0] = ang * 180/np.pi/2
    hsv[:, :, 1] = cv2.normalize(mag ,None ,0, 255, cv2.NORM_MINMAX)

    return hsv

def generate_opticalFlow_data(dataset_dir):
    # Create training data folder
    save_dir = './optical_flow_dir'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for cam in ['cam1', 'cam2']:
        save_dir_cam = os.path.join(save_dir, cam)
        if not os.path.exists(save_dir_cam):
            os.makedirs(save_dir_cam)

        # List all persons in the folder
        all_persons = sorted(os.listdir(os.path.join(dataset_dir, cam)))

        for person in all_persons:
            save_dir_cam_person = os.path.join(save_dir_cam, person)
            if not os.path.exists(save_dir_cam_person):
                os.makedirs(save_dir_cam_person)

            print('Computing optical flow for person: ', person)
            all_images_person_path = os.path.join(dataset_dir, cam, person)
            all_images_person = os.listdir(all_images_person_path)
            all_images_person = sorted(all_images_person)

            # Compute optical flow
            for i in tqdm(range(len(all_images_person))):
                if i == len(all_images_person)-1:
                    image_path = os.path.join(dataset_dir, cam, person, all_images_person[i-1])

                    if image_path[-3:] == 'png':
                        prev_frame = cv2.imread(image_path)
                        prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

                        image_path = os.path.join(dataset_dir, cam, person, all_images_person[i])

                        if image_path[-3:] == 'png':
                            curr_frame_1 = cv2.imread(image_path)
                            curr_frame = cv2.cvtColor(curr_frame_1, cv2.COLOR_BGR2GRAY)
                            # Compute Optical Flow
                            optFlw = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame,\
                                                         None, 0.5, 3, 5, 3, 7, 1.2, 1)
                            # Generate Optical Flow Image
                            optFlw_img = optical_flow_image_gen(optFlw, curr_frame_1)
                            image_save_path = os.path.join(save_dir_cam_person, all_images_person[i])
                            cv2.imwrite(image_save_path, optFlw_img)

                else:
                    image_path = os.path.join(dataset_dir, cam, person, all_images_person[i])

                    if image_path[-3:] == 'png':
                        prev_frame = cv2.imread(image_path)
                        prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

                        image_path = os.path.join(dataset_dir, cam, person, all_images_person[i+1])

                        if image_path[-3:] == 'png':
                            curr_frame_1 = cv2.imread(image_path)
                            curr_frame = cv2.cvtColor(curr_frame_1, cv2.COLOR_BGR2GRAY)
                            # Compute Optical Flow
                            optFlw = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame,\
                                                         None, 0.5, 3, 5, 3, 7, 1.2, 1)
                            # Generate Optical Flow Image
                            optFlw_img = optical_flow_image_gen(optFlw, curr_frame_1)
                            image_save_path = os.path.join(save_dir_cam_person, all_images_person[i])
                            cv2.imwrite(image_save_path, optFlw_img)


# Returns shuffled data
def generate_random_permutation(data_root, train_test_split):
    nPerson_cam1 = os.listdir(data_root + '/cam1')
    nPerson_cam2 = os.listdir(data_root + '/cam2')

    try:
        assert len(nPerson_cam1) == len(nPerson_cam2)
    except AssertionError:
        print('Un-equal camera 1 and camera 2 persons!')
        raise

    nPerson_cam1 = sorted(nPerson_cam1)
    random.shuffle(nPerson_cam1)

    split = int(len(nPerson_cam1)*train_test_split)

    train_set = nPerson_cam1[0:split]
    test_set  = nPerson_cam1[split:]

    print('No. of training set : ', len(train_set))
    print('No. of testing set  : ', len(test_set))
    return train_set, test_set


#################################### Main #####################################
# Create dataset folder if it does not exist
if not os.path.exists('./dataset'):
    os.makedirs('./dataset')

# Generate optical flow
if args.gen_opt_flow == 'True':
    generate_opticalFlow_data(args.dataset_dir)

if args.train_test_split > 0:
    # Radomize data
    train_data, test_data = generate_random_permutation(args.dataset_dir, args.train_test_split)

    # Save training data
    save_file = '%s/%s_train.txt' % ('./dataset', args.data_name)
    f = open(save_file, 'w')
    for data in train_data:
        f.write('%s' % (data))
        f.write('\n')
    f.close()

    # Save testing data
    save_file = '%s/%s_test.txt' % ('./dataset', args.data_name)
    f = open(save_file, 'w')
    for data in test_data:
        f.write('%s' % (data))
        f.write('\n')
    f.close()
#-------------------------------------------------------------------------------
