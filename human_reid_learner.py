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
        identity_loss_l = nn.NLLLoss(reduce=True)
        identity_loss_r = nn.NLLLoss(reduce=True)
        pairwise_distance = nn.PairwiseDistance(p=2)
        HingeEmbeddLoss = nn.HingeEmbeddingLoss(margin=opt.margin, reduce=True)

        # Build graph
        if opt.use_opt_flow == True:
            net = Net(5, opt.hidden_size, opt.nPersons, opt.drop)
        else:
            net = Net(3, opt.hidden_size, opt.nPersons, opt.drop)
        # Set model to train
        net.train()

        # GPU
        if opt.use_gpu:
            net = net.cuda()
        else:
            print('GPU unavailable!')

        # Optimizer
        optimizer = optim.SGD(net.parameters(), lr=opt.l_rate,  weight_decay=opt.l2, momentum=0.9)

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
            interm_loss = 0.0
            for i in range(opt.nPersons*2):
                if i%2 == 0:
                    output_feed = train_dataLoader.gen_data_batch(opt.batch_size, int(i/2), 1, data_augmentation=True)
                    l_batch, r_batch, p_n_batch, l_idx_batch, r_idx_batch = output_feed
                else:
                    output_feed = train_dataLoader.gen_data_batch(opt.batch_size, None, -1, data_augmentation=True)
                    l_batch, r_batch, p_n_batch, l_idx_batch, r_idx_batch = output_feed

                # Input data
                if opt.use_gpu:
                    # Convert to GPU
                    l_batch     = torch.tensor(l_batch,     dtype=torch.float).cuda()
                    r_batch     = torch.tensor(r_batch,     dtype=torch.float).cuda()
                    l_idx_batch = torch.tensor(l_idx_batch, dtype=torch.long).cuda()
                    r_idx_batch = torch.tensor(r_idx_batch, dtype=torch.long).cuda()
                    p_n_batch   = torch.tensor(p_n_batch,   dtype=torch.long).cuda()
                else:
                    l_batch     = torch.tensor(l_batch,     dtype=torch.float)
                    r_batch     = torch.tensor(r_batch,     dtype=torch.float)
                    l_idx_batch = torch.tensor(l_idx_batch, dtype=torch.long)
                    r_idx_batch = torch.tensor(r_idx_batch, dtype=torch.long)
                    p_n_batch   = torch.tensor(p_n_batch,   dtype=torch.long)

                # Convert to Variable
                l_batch = Variable(l_batch)
                r_batch = Variable(r_batch)
                p_n_batch   = Variable(p_n_batch)
                l_idx_batch = Variable(l_idx_batch)
                r_idx_batch = Variable(r_idx_batch)

                # Forward pass
                l_output, r_output, l_id, r_id, hidden = net(l_batch, r_batch, opt.sequence_length, hidden=None)

                # Compute loss
                l2_dist = pairwise_distance(l_output, r_output)
                loss = HingeEmbeddLoss(l2_dist, p_n_batch) + identity_loss_l(l_id, l_idx_batch) + identity_loss_r(r_id, r_idx_batch)

                # Run Optim
                loss.backward()

                # Clip gradient
                torch.nn.utils.clip_grad_norm_(net.parameters(), opt.clip_grad)

                # Update the weights
                optimizer.step()

                # Accumlate loss
                interm_loss += loss.clone().cpu().data.numpy()

            if eph % opt.summary_freq == 0:
                interm_loss = interm_loss/(opt.nPersons*2)
                writer.add_scalar('loss', interm_loss, eph)
                for name, param in net.named_parameters():
                    writer.add_histogram(name, param.clone().cpu().data.numpy(), eph)
                print('Loss: ', np.round(interm_loss, 3), ' at Epoch: ', str(eph + 1))

            if eph % opt.save_latest_freq == 0:
                model_name = 'hnRiD_' + str(eph) + '.ckpt'
                checkpoint_path = os.path.join(opt.checkpoint_dir, model_name)
                torch.save(net.state_dict(), checkpoint_path)
                print("Intermediate file saved")

        model_name = 'hnRiD_latest.ckpt'
        checkpoint_path = os.path.join(opt.checkpoint_dir, model_name)
        torch.save(net.state_dict(), checkpoint_path)
        print("Training Complete and latest checkpoint saved!")

        # Export scalar data to JSON for external processing
        writer.export_scalars_to_json(str(opt.exp_name) + ".json")
        writer.close()
