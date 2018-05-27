from __future__ import print_function
from data_Loader import *
from nets import *
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter

class reid_Learner(object):
    def __init__(self):
        pass

    def train(self, opt):
        # Load Flags
        self.opt = opt

        # Tensorboard Writer
        writer = SummaryWriter()

        # Load the data
        train_dataLoader = dataLoader(opt.directory, opt.train_test_split_dir,\
                                      opt.nPersons, opt.dataset_name,\
                                      opt.image_height, opt.image_width,\
                                      use_opt_flow=opt.use_opt_flow,\
                                      n_steps=opt.sequence_length,\
                                      optical_flow_dir=opt.opt_flow_dir,\
                                      mode='Train')

        # Loss functions
        identity_loss     = nn.CrossEntropyLoss(reduce=True)
        pairwise_distance = nn.PairwiseDistance(p=2)
        HingeEmbeddLoss   = nn.HingeEmbeddingLoss(margin=opt.margin, reduce=True)

        # Build graph
        if opt.use_opt_flow == True:
            net = Net(5, opt.hidden_size, opt.drop)
        else:
            net = Net(3, opt.hidden_size, opt.drop)
        # Set model to train
        net.train()

        # GPU
        net = net.cuda()

        # Optimizer
        optimizer = optim.Adam(net.parameters(), lr=opt.l_rate, eps=1e-08, weight_decay=opt.l2)

        # Check if training has to be continued
        if opt.continue_train:
            if opt.init_checkpoint_file is None:
                print('Enter a valid checkpoint file')
            else:
                load_model = os.path.join(opt.checkpoint_dir, opt.init_checkpoint_file)
            print("Resume training from previous checkpoint: %s" % opt.init_checkpoint_file)
            net.load_state_dict(torch.load(load_model))

        # Begin Training
        for eph in range(opt.start_step, opt.max_steps):
            for i in range(opt.nPersons*2):
                if i%2 == 0:
                    output_feed = train_dataLoader.gen_data_batch(opt.batch_size, int(i/2), 1, data_augmentation=True)
                    l_batch, r_batch, p_n_batch, l_idx_batch, r_idx_batch = output_feed
                else:
                    output_feed = train_dataLoader.gen_data_batch(opt.batch_size, None, -1, data_augmentation=True)
                    l_batch, r_batch, p_n_batch, l_idx_batch, r_idx_batch = output_feed

                # Input data
                l_batch     = torch.tensor(l_batch,  dtype=torch.float).cuda()
                r_batch     = torch.tensor(r_batch,  dtype=torch.float).cuda()
                p_n_batch   = torch.tensor(p_n_batch, dtype=torch.long).cuda()
                l_idx_batch = torch.tensor(l_idx_batch, dtype=torch.long).cuda()
                r_idx_batch = torch.tensor(r_idx_batch, dtype=torch.long).cuda()

                l_batch = Variable(l_batch)
                r_batch = Variable(r_batch)
                p_n_batch   = Variable(p_n_batch)
                l_idx_batch = Variable(l_idx_batch)
                r_idx_batch = Variable(r_idx_batch)

                # Forward pass
                l_output, r_output = net(l_batch, r_batch, steps=opt.sequence_length)

                # Compute loss
                l2_dist = pairwise_distance(l_output, r_output)
                loss = HingeEmbeddLoss(l2_dist, p_n_batch) + identity_loss(l_output, l_idx_batch) + identity_loss(r_output, r_idx_batch)

                # Run Optim
                loss.backward()

                # Clip gradient
                torch.nn.utils.clip_grad_norm(net.parameters(), opt.clip_grad)

                # Update the weights
                optimizer.step()

                if i % opt.summary_freq == 0:
                    interm_loss = loss.clone().cpu().data.numpy()
                    writer.add_scalar('loss', interm_loss, i)
                    for name, param in net.named_parameters():
                        writer.add_histogram(name, param.clone().cpu().data.numpy(), i)
                    print('Loss: ', interm_loss, ' at Epoch: ', str(eph + 1), ' iteration: ', i)

            if eph % opt.save_latest_freq == 0:
                model_name = 'hnRiD_' + str(i) + '.ckpt'
                checkpoint_path = os.path.join(opt.checkpoint_dir, model_name)
                torch.save(net.state_dict(), checkpoint_path)
                print("Intermediate file saved")

        model_name = 'hnRiD_latest.ckpt'
        checkpoint_path = os.path.join(opt.checkpoint_dir, model_name)
        torch.save(net.state_dict(), checkpoint_path)
        print("Training Complete and latest checkpoint saved!")

        # Export scalar data to JSON for external processing
        writer.export_scalars_to_json("./logs/" + str(opt.exp_name) + ".json")
        writer.close()
