import matplotlib.pyplot as plt
from data_Loader import dataLoader

data_directory = 'C:/Users/k-any/Uoit/Human-Re-id/Dataset/iLIDS-VID/i-LIDS-VID/sequences'
train_test_dir = './dataset'
dataset_name   = 'dataset_1_train1.txt'
nPersons = 150
optical_flow_dir = './optical_flow_dir'
image_height = 64
image_width  = 48
mode='Train'
use_opt_flow=False

dl = dataLoader(data_directory, train_test_dir, 2, dataset_name,
                image_height, image_width, use_opt_flow,n_steps=16, optical_flow_dir=optical_flow_dir, mode='Train')
output_list = dl.gen_data_batch(1, 0, 1, True)

p1_batch, p2_batch, p_n_batch, p1_idx_batch, p2_idx_batch = output_list
print('p1', p1_batch.shape)
print('p2', p2_batch.shape)
print('p_n', p_n_batch.shape)
print('p1_', p1_idx_batch.shape)
print('p2_', p2_idx_batch.shape)

#    
#    
#if use_opt_flow ==  True:
#    output_list = next(train_gen)
#    p1_batch, p2_batch, p1_opt_batch, p2_opt_batch, p_n_batch, p1_idx_batch, p2_idx_batch = output_list
#    print(p1_batch.shape)
#    print(p2_batch.shape)
#    print(p1_opt_batch.shape)
#    print(p2_opt_batch.shape)
#    print(p_n_batch.shape)
#    print(p1_idx_batch[0])
#    print(p2_idx_batch[0])
#        
#    for i in range(1, 17):
##        plt.subplot(8, 8, i)
##        plt.imshow(abs(p1_batch[0, i-1, :, :, :])*255)
##        plt.axis('off')
##        plt.subplot(8, 8, i+16)
##        plt.imshow(abs(p1_opt_batch[0, i-1, :, :, :])*255)
##        plt.axis('off')
##        plt.subplot(8, 8, i+32)
##        plt.imshow(abs(p2_batch[0, i-1, :, :, :])*255)
##        plt.axis('off')
##        plt.subplot(8, 8, i+48)
##        plt.imshow(abs(p2_opt_batch[0, i-1, :, :, :])*255)
##        plt.axis('off')
#
#        plt.subplot(8, 8, i)
#        plt.imshow(abs(p1_batch[0, i-1, :, :, :]))
#        plt.axis('off')
#        plt.subplot(8, 8, i+16)
#        plt.imshow(abs(p1_opt_batch[0, i-1, :, :, :]))
#        plt.axis('off')
#        plt.subplot(8, 8, i+32)
#        plt.imshow(abs(p2_batch[0, i-1, :, :, :]))
#        plt.axis('off')
#        plt.subplot(8, 8, i+48)
#        plt.imshow(abs(p2_opt_batch[0, i-1, :, :, :]))
#        plt.axis('off')
#else:
#    p1_batch, p2_batch, p_n_batch, p1_idx_batch, p2_idx_batch = next(train_gen)
#    for i in range(1, 17):
#        plt.subplot(4, 8, i)
#        plt.imshow(p1_batch[1, i-1, :, :, :])
#        plt.axis('off')
#        plt.subplot(4, 8, i+16)
#        plt.imshow(p2_batch[1, i-1, :, :, :])
#        plt.axis('off')

if p_n_batch[0] == 1.0:
    print('same')
else:
    print('Not same')

plt.show()
