import torch
from torch import nn
import lightning as L
import torch.nn.functional as F

class MNRE(  nn.Module ):

    def __init__(self, n_data, n_params, embed_size, data_hlayer_size, tail_hlayer_size):
        '''
        n_data: number of data points in an instance of the observed data
        n_params: total number of parameters to be inferred
        embed_size: size of the embedding layer for the data
        data_hlayer_size: size of the hidden layer in the data processor
        tail_hlayer_size: size of the hidden layer in the tails
        '''
        super(MNRE, self).__init__()
        self.n_data = n_data
        self.n_params = n_params

        self.data_processor = nn.Sequential(nn.Linear(n_data, data_hlayer_size), nn.ReLU(),  nn.Linear(data_hlayer_size, embed_size)) # a fully connected NN 

        # each tail is a MLP that evaluates one of the marginal ratios
        self.tail_1 = nn.Sequential(
            nn.Linear(embed_size+1, tail_hlayer_size),
            nn.ReLU(),
            nn.Linear(tail_hlayer_size, 1)
        )
        self.tail_2 = nn.Sequential(
            nn.Linear(embed_size+1, tail_hlayer_size),
            nn.ReLU(),
            nn.Linear(tail_hlayer_size, 1)
        )
        self.tail_3 = nn.Sequential(
            nn.Linear(embed_size+1, tail_hlayer_size),
            nn.ReLU(),
            nn.Linear(tail_hlayer_size, 1)
        )
        self.tail_12 = nn.Sequential( 
            nn.Linear(embed_size+2, tail_hlayer_size),
            nn.ReLU(),
            nn.Linear(tail_hlayer_size, 1)
        )
        self.tail_13 = nn.Sequential(
            nn.Linear(embed_size+2, tail_hlayer_size),
            nn.ReLU(),
            nn.Linear(tail_hlayer_size, 1)
        )
        self.tail_23 = nn.Sequential(
            nn.Linear(embed_size+2, tail_hlayer_size),
            nn.ReLU(),
            nn.Linear(tail_hlayer_size, 1)
        )

    def forward(self, data, theta):
        # data: (batch_size, n_data)
        # theta: (batch_size, n_params)

        data_embed = self.data_processor(data)
        lnr_1 = self.tail_1(torch.cat((data_embed, theta[:, 0].unsqueeze(1)), dim=1))
        lnr_2 = self.tail_2(torch.cat((data_embed, theta[:, 1].unsqueeze(1)), dim=1))
        lnr_3 = self.tail_3(torch.cat((data_embed, theta[:, 2].unsqueeze(1)), dim=1))
        lnr_12 = self.tail_12(torch.cat((data_embed, theta[:, 0].unsqueeze(1), theta[:, 1].unsqueeze(1)), dim=1))
        lnr_13 = self.tail_13(torch.cat((data_embed, theta[:, 0].unsqueeze(1), theta[:, 2].unsqueeze(1)), dim=1))
        lnr_23 = self.tail_23(torch.cat((data_embed, theta[:, 1].unsqueeze(1), theta[:, 2].unsqueeze(1)), dim=1))

        return torch.cat((lnr_1, lnr_2, lnr_3, lnr_12, lnr_13, lnr_23), dim=1)
    
    def marginal_ratio_1(self, data, theta):
        data_embed = self.data_processor.forward(data)
        lnr_1 = self.tail_1.forward(torch.cat((data_embed, theta), dim=1))
        return lnr_1
    def marginal_ratio_2(self, data, theta):
        data_embed = self.data_processor.forward(data)
        lnr_2 = self.tail_2.forward(torch.cat((data_embed, theta), dim=1))
        return lnr_2
    def marginal_ratio_3(self, data, theta):
        data_embed = self.data_processor.forward(data)
        lnr_3 = self.tail_3.forward(torch.cat((data_embed, theta), dim=1))
        return lnr_3
    def marginal_ratio_12(self, data, theta):
        data_embed = self.data_processor.forward(data)
        lnr_12 = self.tail_12.forward(torch.cat((data_embed, theta), dim=1))
        return lnr_12
    def marginal_ratio_23(self, data, theta):
        data_embed = self.data_processor.forward(data)
        lnr_23 = self.tail_23.forward(torch.cat((data_embed, theta), dim=1))
        return lnr_23
    def marginal_ratio_13(self, data, theta):
        data_embed = self.data_processor.forward(data)
        lnr_13 = self.tail_13.forward(torch.cat((data_embed, theta), dim=1))
        return lnr_13
    

class lightning_MNRE ( L.LightningModule ):
    def __init__(self, model: MNRE, optimizer: str = 'adam', lr: float = 1e-3, weight_decay: float = 1e-5):
        super(lightning_MNRE, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters()

    def configure_optimizers(self):
        if self.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")
        return optimizer
    
    def training_step(self, batch, batch_idx):
        '''
        batch: a list containing observed data, joint parameters and scrambled parameters
        batch_idx: index of the batch
        '''
        data, theta_joint, theta_scrambled = batch
        # the second half of all_data is the copy of the first half
        all_data = torch.cat((data, data), dim=0)
        # the first half of all_params contains parameters from the joint, associated with the first half of all_data
        # the second half of all_params contains parameters from the marginal, independent of the second half of all_data
        all_params = torch.cat((theta_joint, theta_scrambled), dim=0)
        logits = self.model(all_data, all_params) # has shape (batch_size*2, 6)
        labels = torch.cat((torch.ones(data.size(0), 6), torch.zeros(data.size(0), 6)), dim=0)
        loss = F.binary_cross_entropy_with_logits(logits, labels,reduction='none') # will have shape (batch_size*2, 6)
        # log the loss for each tail
        self.log('train_loss_1', loss[:, 0].mean(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_loss_2', loss[:, 1].mean(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_loss_3', loss[:, 2].mean(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_loss_12', loss[:, 3].mean(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_loss_13', loss[:, 4].mean(), on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_loss_23', loss[:, 5].mean(), on_step=False, on_epoch=True, prog_bar=True)
        # log the total loss
        loss = loss.mean() # mean over all tails
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    

    def validation_step(self, batch, batch_idx):
        data, theta_joint, theta_scrambled = batch
        all_data = torch.cat((data, data), dim=0)
        all_params = torch.cat((theta_joint, theta_scrambled), dim=0)
        logits = self.model(all_data, all_params)
        labels = torch.cat((torch.ones(data.size(0), 6), torch.zeros(data.size(0), 6)), dim=0)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        accuracy = (torch.sigmoid(logits) > 0.5).float().eq(labels).sum() / labels.numel()
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss, accuracy
    
    