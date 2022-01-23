import os

import torch


class ModelHelper(object):

    @staticmethod
    def save_state_dict(policy_net, model_save_path):
        torch.save(policy_net.state_dict(), model_save_path)

    @staticmethod
    def load_state_dict(policy_net, model_pth, map_location):
        state_dict = torch.load(model_pth, map_location=map_location)
        policy_net.load_state_dict(state_dict)

    @staticmethod
    def soft_update_of_target_network(local_model, target_model, tau):
        """Updates the target network in the direction of the local network but by taking a step size
        less than one so the target network's parameter values trail the local networks. This helps stabilise training"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    @staticmethod
    def skipping_step_update_of_target_network(q_network_local, q_network_target, global_step_number,
                                               update_every_n_steps):
        """Updates the target network every n steps"""
        if global_step_number % update_every_n_steps == 0:
            q_network_target.load_state_dict(q_network_local.state_dict())

    @staticmethod
    def copy_model_over(from_model, to_model):
        """Copies model parameters from from_model to to_model"""
        for to_model, from_model in zip(to_model.parameters(), from_model.parameters()):
            to_model.data.copy_(from_model.data.clone())

    @staticmethod
    def store_model_optimizer(model_dict, optimizer_dict, model_sv_folder, prefix, i_episode):
        for key, model in model_dict.items():
            model_name = prefix + "_" + key + "_%d.mdl" % (i_episode + 1)
            model_pth = os.path.join(model_sv_folder, model_name)
            print("Trained model is stored to path : {}".format(model_pth))

            torch.save(model.state_dict(), model_pth)
        # for key, optimizer in optimizer_dict:
        #     optimizer_name = prefix + "_" + key + "_%d.optimizer" % (i_episode + 1)
        #     optimizer_pth = os.path.join(model_sv_folder, optimizer_name)
        #     print("Trained model is loaded from path : {}".format(optimizer_pth))

    @staticmethod
    def load_model_optimizer(model_dict, optimizer_dict, model_ld_folder, predix, index, device):
        for key, model in model_dict.items():
            model_name = predix + "_" + key + "_%d.mdl" % index
            # model_name = "Agent_dqn_state_dict" + "_%d.mdl" % index

            model_pth = os.path.join(model_ld_folder, model_name)
            if not os.path.exists(model_pth):
                print("Path to trained model to be loaded does not exist | Path : {}".format(model_pth))
                raise FileNotFoundError
            ModelHelper.load_state_dict(policy_net=model, model_pth=model_pth, map_location=device)
            print("Trained model is loaded from path : {}".format(model_pth))
