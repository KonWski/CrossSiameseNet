import torch
import logging
import os

def save_checkpoint(checkpoint: dict, checkpoint_path: str):
    '''
    saves checkpoint on given checkpoint_path
    '''

    model_dir = os.path.dirname(checkpoint_path)
    if not os.path.isdir(model_dir):
        logging.info(f"Creating a missing directory: {model_dir}")
        os.mkdir(model_dir)

    torch.save(checkpoint, checkpoint_path)

    logging.info(8*"-")
    logging.info(f"Saved model to checkpoint: {checkpoint_path}")
    logging.info(f"Epoch: {checkpoint['epoch']}")
    logging.info(8*"-")



def load_checkpoint(model, checkpoint_path: str):
    '''
    loads model checkpoint from given path

    Parameters
    ----------
    model
        loaded type of model

    Notes
    -----
    checkpoint: dict
                parameters retrieved from training process i.e.:
                - model_state_dict
                - last finished number of epoch
                - save time
                - loss from last epoch testing
                
    '''
    checkpoint = torch.load(checkpoint_path)

    # load parameters from checkpoint
    model.load_state_dict(checkpoint["model_state_dict"])

    # print loaded parameters
    logging.info(f"Loaded model from checkpoint: {checkpoint_path}")

    for param, param_value in checkpoint.items():

        if param != "model_state_dict":
            logging.info(f"{param}: {param_value}")

    logging.info(8*"-")

    return model, checkpoint