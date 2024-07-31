import torch
import logging
from CrossSiameseNet.SiameseMolNet import  SiameseMolNet
from CrossSiameseNet.SiameseMolNetTL import  SiameseMolNetTriplet


def save_checkpoint(checkpoint: dict, checkpoint_path: str):
    '''
    saves checkpoint on given checkpoint_path
    '''
    torch.save(checkpoint, checkpoint_path)

    logging.info(8*"-")
    logging.info(f"Saved model to checkpoint: {checkpoint_path}")
    logging.info(f"Epoch: {checkpoint['epoch']}")
    logging.info(8*"-")


def load_checkpoint(checkpoint_path: str, triple_loss: bool = False):
    '''
    loads model checkpoint from given path

    Parameters
    ----------
    checkpoint_path : str
        Path to checkpoint

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
    cf_size = 2048

    # initiate model
    if triple_loss:
        model = SiameseMolNetTriplet
    else:
        model = SiameseMolNet(cf_size)

    # load parameters from checkpoint
    model.load_state_dict(checkpoint["model_state_dict"])

    # print loaded parameters
    logging.info(f"Loaded model from checkpoint: {checkpoint_path}")
    logging.info(f"Dataset: {checkpoint['dataset']}")    
    logging.info(f"Epoch: {checkpoint['epoch']}")
    logging.info(f"Save dttm: {checkpoint['save_dttm']}")
    logging.info(f"Train loss: {checkpoint['train_loss']}")    
    logging.info(f"Test loss: {checkpoint['test_loss']}")

    logging.info(8*"-")

    return model, checkpoint