import os
import cv2
import glob
import random
import numpy as np

class dataLoader(object):
    def __init__(self, data_directory, train_test_dir, nPersons, dataset_name,
                 image_height, image_width, use_opt_flow=False, n_steps=16,
                 optical_flow_dir=None, mode='Train'):
        self.mode = mode
        self.n_steps = n_steps
        self.nPersons = nPersons
        self.use_opt_flow = use_opt_flow
        self.image_width  = image_width
        self.image_height = image_height
        self.dataset_name = dataset_name
        self.data_directory = data_directory
        self.train_test_dir = train_test_dir
        self.optical_flow_dir = optical_flow_dir
        self.get_data()

    def get_data(self):
        all_person = []
        # Full images file path
        file_path = os.path.join('./dataset', self.dataset_name)

        # Get all the test persons
        with open(file_path, 'r') as f:
            all_data = f.readlines()
        f.close()

        for data in all_data:
            if data == '\n':
                continue
            # Removes '\n' at the begining
            person = data[:-1]
            all_person.append(person)

        # Get all persons
        all_person = sorted(all_person)
        nPersons   = len(all_person)
        all_seq_cam1 = []
        all_seq_cam2 = []

        for i in range(nPersons):
            person_cam1_path = os.path.join(self.data_directory, 'cam1', all_person[i])
            person_cam2_path = os.path.join(self.data_directory, 'cam2', all_person[i])
            all_images_person_cam1 = sorted(glob.glob(person_cam1_path + "/*.png"))
            all_images_person_cam2 = sorted(glob.glob(person_cam2_path + "/*.png"))

            interm_person_cam1 = []
            for j in range(len(all_images_person_cam1)):
                person_1 = all_images_person_cam1[j]
                interm_person_cam1.append(person_1)
            interm_person_cam2 = []
            for k in range(len(all_images_person_cam2)):
                person_2 = all_images_person_cam2[k]
                interm_person_cam2.append(person_2)

            all_seq_cam1.append(interm_person_cam1)
            all_seq_cam2.append(interm_person_cam2)

        print('Total People: ', len(all_person))
        print('Seq: ', len(all_seq_cam1))

        self.all_person = all_person
        self.all_seq_cam1 = all_seq_cam1
        self.all_seq_cam2 = all_seq_cam2

    def reduce_mean_and_std(self, image):
        H, W, C = image.shape
        for i in range(C):
            v = image[:, :, i].std(ddof=1)
            m = image[:, :, i].mean()
            image[:, :, i] = (image[:, :, i] - m) / v

        return image

    def imageCrop(self, image, shiftx, shifty):
        in_h, in_w, in_c = image.shape
        width_offset  = int(shiftx)
        height_offset = int(shifty)

        startx = width_offset
        endx   = width_offset + 40
        starty = height_offset
        endy   = height_offset + 56

        cropped_img = image[starty:endy, startx:endx]

        # Reduce Mean
        #cropped_img = cropped_img - cropped_img.mean()

        return cropped_img

    def gen_data_batch(self, batch_size, nPerson, pos_neg, data_augmentation=False):
        # Generate data based on training/validation
        p1_batch  = []
        p2_batch  = []
        p_n_batch = []
        p1_idx_batch = []
        p2_idx_batch = []
        # Generate training batch
        for _ in range(batch_size):
            p1_step = []
            p2_step = []

            if pos_neg == 1:
                # Positive sample
                actualSampleSeqLen = self.n_steps
                nSeqA = len(self.all_seq_cam1[nPerson])
                nSeqB = len(self.all_seq_cam2[nPerson])
                # Check for un-even sequences
                if nSeqA <= self.n_steps or nSeqB <= self.n_steps:
                    if nSeqA < nSeqB:
                        actualSampleSeqLen = nSeqA
                    else:
                        actualSampleSeqLen = nSeqB
                # StartA and startB
                startA = np.random.randint(0, nSeqA - actualSampleSeqLen, 1)[0]
                startB = np.random.randint(0, nSeqB - actualSampleSeqLen, 1)[0]
                # Persons index
                pA_person = nPerson
                pB_person = nPerson
            #---------------------------------------------------------------
            else:
                # Negative sample
                permAllPersons = list(range(len(self.all_person)))
                random.shuffle(permAllPersons)
                personA = permAllPersons[0]
                personB = permAllPersons[1]
                actualSampleSeqLen = self.n_steps
                nSeqA = len(self.all_seq_cam1[personA])
                nSeqB = len(self.all_seq_cam2[personB])
                # Check for un-even sequences
                if nSeqA <= self.n_steps or nSeqB <= self.n_steps:
                    if nSeqA < nSeqB:
                        actualSampleSeqLen = nSeqA
                    else:
                        actualSampleSeqLen = nSeqB
                # StartA and startB
                startA = np.random.randint(0, nSeqA - actualSampleSeqLen, 1)[0]
                startB = np.random.randint(0, nSeqB - actualSampleSeqLen, 1)[0]
                # Persons index
                pA_person = personA
                pB_person = personB
            #---------------------------------------------------------------
            # Read the images
            for step in range(self.n_steps):
                # Path
                rgb_image_path_1 = os.path.join(self.all_seq_cam1[pA_person][startA + step])
                rgb_image_path_2 = os.path.join(self.all_seq_cam2[pB_person][startB + step])
                # Read RGB image
                rgb_image_1 = cv2.imread(rgb_image_path_1)
                rgb_image_2 = cv2.imread(rgb_image_path_2)
                # Rescale images
                H_l, W_l, C_l = rgb_image_1.shape
                H_r, W_r, C_r = rgb_image_2.shape
                if H_l != self.image_height and W_l != self.image_width:
                    rgb_image_1 = cv2.resize(rgb_image_1, (self.image_width, self.image_height))
                if H_r != self.image_height and W_r != self.image_width:
                    rgb_image_2 = cv2.resize(rgb_image_2, (self.image_width, self.image_height))
                # Convert to RGB space
                rgb_image_1 = cv2.cvtColor(rgb_image_1, cv2.COLOR_BGR2YUV)
                rgb_image_2 = cv2.cvtColor(rgb_image_2, cv2.COLOR_BGR2YUV)
                # Collect RGB frames
                p1_step.append(rgb_image_1)
                p2_step.append(rgb_image_2)
            #---------------------------------------------------------------
            # Augmentation
            if data_augmentation == True:
                # Random Flip
                doflip_A = np.random.randint(0, 2, 1)[0]
                doflip_B = np.random.randint(0, 2, 1)[0]
                # Random Crop
                crpxA = np.random.randint(1, 9, 1)[0]
                crpyA = np.random.randint(1, 9, 1)[0]
                crpxB = np.random.randint(1, 9, 1)[0]
                crpyB = np.random.randint(1, 9, 1)[0]

            for i in range(len(p1_step)):
                rgb_image_1 = p1_step[i]
                rgb_image_2 = p2_step[i]
                # Normalize each channel in the image
                rgb_image_1 = self.reduce_mean_and_std(rgb_image_1)
                rgb_image_2 = self.reduce_mean_and_std(rgb_image_2)
                # Check to do Augmentation
                if data_augmentation == True:
                    # Flip
                    if doflip_A == 0:
                        rgb_image_1 = cv2.flip(rgb_image_1, 1)
                    if doflip_B == 0:
                        rgb_image_2 = cv2.flip(rgb_image_2, 1)
                    # Crop
                    rgb_image_1 = self.imageCrop(rgb_image_1, crpxA, crpyA)
                    rgb_image_2 = self.imageCrop(rgb_image_2, crpxB, crpyB)
                # Change the ordering- nChannel first
                rgb_image_1 = np.transpose(rgb_image_1, axes=[2, 0, 1])
                rgb_image_2 = np.transpose(rgb_image_2, axes=[2, 0, 1])
                # Save the latest
                p1_step[i] = rgb_image_1
                p2_step[i] = rgb_image_2
            #---------------------------------------------------------------
            # Collect all --n_steps, H, W, C
            p1_step = np.array(p1_step)
            p2_step = np.array(p2_step)

            # Person 1 and 2
            p1_batch.append(p1_step)
            p2_batch.append(p2_step)

            p_n_batch.append(pos_neg)

            p1_idx_batch.append(pA_person)
            p2_idx_batch.append(pB_person)
            #---------------------------------------------------------------

        output_list = [np.array(p1_batch),  np.array(p2_batch,),\
                       np.array(p_n_batch), np.array(p1_idx_batch),\
                       np.array(p2_idx_batch)]

        return output_list
