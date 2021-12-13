import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        
        # use a pretrained model: resmet
        resnet = models.resnet50(pretrained=True)
        # set the parameters untrainable
        for param in resnet.parameters():
            param.requires_grad_(False)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        # fully connected layer as an embedding
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        # feed into a pretrained layer
        features = self.resnet(images)
        
        # flatten the input
        features = features.view(features.size(0), -1)
        
        # embed the image
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        # defining model sizes
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # embedding layer
        self.embed = nn.Embedding(vocab_size, embed_size, num_layers)
        
        # lstm layer
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)

        # fully connected layer
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions):
        # removing <end> token
        captions = captions[:, :-1]
        
        # embedding
        captions_embeded = self.embed(captions)
        
        # preparing for LSTM
        concatenate = torch.cat((features.unsqueeze(1), captions_embeded), dim=1)
        
        # LSTM
        lstm_out, _ = self.lstm(concatenate)
        
        # Output
        output = self.linear(lstm_out)

        return output

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        outputs = []   
        output_length = 0
        
        while (output_length != max_len+1):
            
            ''' LSTM layer '''
            # input  : (1,1,embed_size)
            # output : (1,1,hidden_size)
            # States should be passed to LSTM on each iteration in order for it to recall the last word it produced.
            output, states = self.lstm(inputs,states)
           
            ''' Linear layer '''
            # input  : (1,hidden_size)
            # output : (1,vocab_size)
            output = self.linear(output.squeeze(dim = 1))
            _, predicted_index = torch.max(output, 1)
            
            # CUDA tensor has to be first converted to cpu and then to numpy.
            # Because numpy doesn't support CUDA ( GPU memory ) directly.
            # See this link for reference : https://discuss.pytorch.org/t/convert-to-numpy-cuda-variable/499
            outputs.append(predicted_index.cpu().numpy()[0].item())
            
            # <end> has index_value = 1 in the vocabulary [ Notebook 1 ]
            # This conditional statement helps to break out of the while loop,
            # as soon as the first <end> is encountered. Length of caption maybe less than 20 at this point.
            if (predicted_index == 1):
                break
            
            # Prepare for net loop iteration 
            # Embed the last predicted word to be the new input of the LSTM
            inputs = self.embed(predicted_index)   
            inputs = inputs.unsqueeze(1)
            
            # Move to the next iteration
            output_length += 1
          
        return outputs