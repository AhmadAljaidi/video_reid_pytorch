from __future__ import print_function
from nets_test import *
import torch
from torch.autograd import Variable
import os
import cv2
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir",     type=str, default='/home/amohamma/UOIT/Human_re_id/iLIDS-VID/i-LIDS-VID/sequences', help="where the dataset is stored")
parser.add_argument("--optflow_dir",     type=str, default='./optical_flow_dir', help="where the optical flow is stored")
parser.add_argument("--test_file",       type=str, default='./dataset/dataset_1_test.txt', help="File containing the test persons")
parser.add_argument("--checkpoint_dir",  type=str, default='./checkpoints', help="Where the ckpt files are")
parser.add_argument("--checkpoint_file", type=str, default='hnRiD_latest', help="checkpoint file name to load")
parser.add_argument("--use_opt_flow",    type=str, default='False', help="True/False")
parser.add_argument("--use_data_aug",    type=str, default='True', help="True/False")
parser.add_argument("--image_height",    type=int, default=64, help="image height")
parser.add_argument("--image_width",     type=int, default=48, help="image width")
parser.add_argument("--n_steps",         type=int, default=16, help="Sequence length")
parser.add_argument("--hidden_size",     type=int, default=128, help="RNN Hidden Size")
parser.add_argument("--use_gpu",         type=int, default=1,   help="Use GPU if available")

args = parser.parse_args()
print('----------------------------------------')
print('FLAGS:')
for arg in vars(args):
    print("'", arg,"'", ": ", getattr(args, arg))
print('----------------------------------------')
print('Testing....')

#-------------------------------------------------------------------------------
def prepare_test_dataset(dataset_dir, dataset_name):
    all_person = []
    # Get all the test persons
    with open(dataset_name, 'r') as f:
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
        person_cam1_path = os.path.join(dataset_dir, 'cam1', all_person[i])
        person_cam2_path = os.path.join(dataset_dir, 'cam2', all_person[i])
        all_images_person_cam1 = sorted(glob.glob(person_cam1_path + "/*.png"))
        all_images_person_cam2 = sorted(glob.glob(person_cam2_path + "/*.png"))

        interm_person_cam1 = []
        for j in range(len(all_images_person_cam1)):
            person_1 = os.path.join('cam1', all_person[i], all_images_person_cam1[j])
            interm_person_cam1.append(person_1)
        interm_person_cam2 = []
        for k in range(len(all_images_person_cam2)):
            person_2 = os.path.join('cam2', all_person[i], all_images_person_cam2[k])
            interm_person_cam2.append(person_2)

        all_seq_cam1.append(interm_person_cam1)
        all_seq_cam2.append(interm_person_cam2)

    return all_person, all_seq_cam1, all_seq_cam2

#-------------------------------------------------------------------------------
def imageCrop(image, shiftx, shifty):
    in_h, in_w, in_c = image.shape
    width_offset  = int(shiftx)
    height_offset = int(shifty)

    startx = width_offset
    endx   = width_offset + 40
    starty = height_offset
    endy   = height_offset + 56

    cropped_img = image[starty:endy, startx:endx]

    # Reduce Mean
    cropped_img = cropped_img - cropped_img.mean()

    return cropped_img

#-------------------------------------------------------------------------------
def reduce_mean_and_std(image):
    image_1 = np.zeros_like(image)
    H, W, C = image.shape
    for i in range(C):
        v = image[:, :, i].std()
        m = image[:, :, i].mean()
        image_1[:, :, i] = (image[:, :, i] - m) / v

    return image_1

#-------------------------------------------------------------------------------
def read_norm_augment_imageSeq(imgSeq, shiftx, shifty, doflip):
    # Read RGB
    rgb_seq = []
    for i in range(len(imgSeq)):
        # Path
        image_path = os.path.join(args.dataset_dir, imgSeq[i])
        # Read RGB image
        rgb_image = cv2.imread(image_path)
        # Rescale images
        H_l, W_l, C_l = rgb_image.shape
        if H_l != args.image_height and W_l != args.image_width:
            rgb_image = cv2.resize(rgb_image, (args.image_width, args.image_height))
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2YUV)
        # Normalize
        rgb_image = reduce_mean_and_std(rgb_image)
        # Data Augmentation
        if args.use_data_aug == 'True':
            if doflip == 0:
                rgb_image = cv2.flip(rgb_image, 1)
            # Crop
            rgb_image = imageCrop(rgb_image, shiftx, shifty)
        # Change the ordering- nChannel first
        rgb_image = np.transpose(rgb_image, axes=[2, 0, 1])
        # Collect RGB frames
        rgb_seq.append(rgb_image)
    # Expand dims -- 1, n_steps, 3, 56, 40
    rgb_seq = np.array(rgb_seq)
    rgb_seq = np.expand_dims(rgb_seq, axis=0)

    return np.array(rgb_seq)
#-------------------------------------------------------------------------------
'''
--------------------------------- MAIN -----------------------------------------
'''
# Get all the images in person folders in sequence
all_person, all_seq_cam1, all_seq_cam2 = prepare_test_dataset(args.dataset_dir, args.test_file)

# Build test graph
if args.use_opt_flow == 'True':
    net = Net(5, args.hidden_size, 1.0)
else:
    net = Net(3, args.hidden_size, 1.0)
# Set model to test
net.eval()

# Use GPU if available
if args.use_gpu == 1:
    net = net.cuda()

# Load checkpoint
load_model = os.path.join(args.checkpoint_dir, args.checkpoint_file)
net.load_state_dict(torch.load(load_model))
print("Checkpoint: %s Loaded!" % args.checkpoint_file)

# Loop through ranks
avgSame  = 0
avgDiff  = 0
avgSameCount = 0
avgDiffCount = 0
nPersons = len(all_person)
sampleSeqLength = int(args.n_steps)

# Initial similarity matrix
simMat = np.zeros((nPersons, nPersons))

for shiftx in range(1, 9):
    print('Shift: ', shiftx)
    shifty = shiftx
    for doflip in range(2):
        #-----------------------------------------------------------------------
        # For cam --1
        feats_cam_a = []
        for i in range(nPersons):
            actualSampleLen = 0
            seqLen = len(all_seq_cam1[i])
            if seqLen > sampleSeqLength:
                actualSampleLen = sampleSeqLength
            else:
                actualSampleLen = seqLen
            # Get the image sequence
            seq = all_seq_cam1[i][0:actualSampleLen]
            imgSeq = read_norm_augment_imageSeq(seq, shiftx, shifty, doflip)
            imgSeq = torch.tensor(imgSeq, dtype=torch.float)
            if args.use_gpu == 1:
                imgSeq = imgSeq.cuda()
            # Net Forward pass
            imgSeq = Variable(imgSeq)
            out = net(imgSeq, steps=sampleSeqLength)
            out = out.clone().cpu().data.numpy()
            feats_cam_a.append(out)
        # Delete
        del imgSeq
        #-----------------------------------------------------------------------
        # For cam --2
        feats_cam_b = []
        for i in range(nPersons):
            actualSampleLen = 0
            seqOffset = 0
            seqLen = len(all_seq_cam2[i])
            if seqLen > sampleSeqLength:
                actualSampleLen = sampleSeqLength
                seqOffset = seqLen - sampleSeqLength
            else:
                actualSampleLen = seqLen
                seqOffset = 0
            # Get the image sequence
            seq = all_seq_cam2[i][seqOffset:seqOffset + actualSampleLen]
            imgSeq = read_norm_augment_imageSeq(seq, shiftx, shifty, doflip)
            imgSeq = torch.tensor(imgSeq, dtype=torch.float)
            if args.use_gpu == 1:
                imgSeq = imgSeq.cuda()
            # Net Forward pass
            imgSeq = Variable(imgSeq)
            out = net(imgSeq, steps=sampleSeqLength)
            out = out.clone().cpu().data.numpy()
            feats_cam_b.append(out)
        # Delete
        del imgSeq
        #-----------------------------------------------------------------------
        # Compute similarity
        for i in range(nPersons):
            for j in range(nPersons):
                fa = feats_cam_a[i]
                fb = feats_cam_b[j]
                dst = np.sqrt(np.sum(np.square(fa - fb)))
                simMat[i][j] = simMat[i][j] + dst
                if i == j:
                    avgSame = avgSame  + dst
                    avgSameCount = avgSameCount + 1
                else:
                    avgDiff = avgDiff + dst
                    avgDiffCount = avgDiffCount + 1
        #-----------------------------------------------------------------------
# Compute CMC
avgSame = avgSame / avgSameCount
avgDiff = avgDiff / avgDiffCount

cmcInds = np.zeros(nPersons)
cmc     = np.zeros(nPersons)
for i in range(nPersons):
    cmcInds[i] = i
    tmp = simMat[i, :]
    o   = np.argsort(tmp)
    # Find the element we want
    indx = 0
    tmpIdx = 0
    for j in range(nPersons):
        if o[j] == i:
            indx = j
    # Compute CMC for each rank
    for j in range(indx, nPersons):
        cmc[j] = cmc[j] + 1

# Compute CMC average
cmc = (cmc / nPersons) * 100
# Show test results
compare_rank = [1, 2, 3, 4, 5]
for i in compare_rank:
    print('Rank CMC percentage for rank = : ', str(i), ' is:', str(cmc[i-1]), ' %')

# Plot CMC
plt.plot(cmcInds, cmc)
plt.title('iLIDS-VID')
plt.xlabel('Rank')
plt.ylabel('CMC(%)')
plt.savefig('test_result.png')
#plt.plot()
