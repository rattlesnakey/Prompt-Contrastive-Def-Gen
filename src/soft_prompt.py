import torch
import torch.nn as nn


class SoftEmbedding(nn.Module):
    def __init__(self, 
                wte: nn.Embedding,
                n_tokens: int = 10, 
                random_range: float = 0.5,
                decoder_start_token_id: int = 2,
                initialize_from_vocab: bool = True):
        """
        Args:
            wte (nn.Embedding): 
            n_tokens (int, optional): number of tokens for task. Defaults to 10.
            random_range (float, optional): range to init embedding (if not initialize from vocab). Defaults to 0.5.
            initialize_from_vocab (bool, optional): initalizes from default vocab. Defaults to True.
        """
        super(SoftEmbedding, self).__init__()
        self.wte = wte
        self.n_tokens = n_tokens
        self.decoder_start_token_id = decoder_start_token_id
        self.learned_embedding = nn.parameter.Parameter(self.initialize_embedding(wte,
                                                                                  n_tokens, 
                                                                                  random_range, 
                                                                                  initialize_from_vocab))

    def initialize_embedding(self, 
                             wte: nn.Embedding,
                             n_tokens: int = 10, 
                             random_range: float = 0.5, 
                             initialize_from_vocab: bool = True):
        """
        Args:
            same as __init__
        Returns:
            torch.float: initialized using original schemes
        """

        if initialize_from_vocab:
            return self.wte.weight[:n_tokens].clone().detach()
        return torch.FloatTensor(n_tokens, wte.weight.size(1)).uniform_(-random_range, random_range)

    def forward(self, tokens):
        """run forward pass
        Args:
            tokens (torch.long): input tokens before encoding
        Returns:
            torch.float: encoding of text concatenated with learned task specifc embedding
        """
        #! decoder
        if tokens[0][0].item() == self.decoder_start_token_id:
            input_prefix_embedding = self.wte(tokens[:, :2])
            input_rest_embedding = self.wte(tokens[:, 2 + self.n_tokens:])
        else:
            #! encoder
            input_prefix_embedding = self.wte(tokens[:, :1])
            input_rest_embedding = self.wte(tokens[:, 1 + self.n_tokens:])
            
            
        learned_embedding = self.learned_embedding.repeat(input_prefix_embedding.size(0), 1, 1)
        if input_rest_embedding.size(1) == 0:
            return torch.cat([input_prefix_embedding, learned_embedding], dim=1)
        else:
            return torch.cat([input_prefix_embedding, learned_embedding, input_rest_embedding], dim=1)


def set_soft_embedding_and_freeze(model, n_tokens):
    s_wte = SoftEmbedding(
        model.get_input_embeddings(), 
        n_tokens=n_tokens, 
        decoder_start_token_id=model.config.decoder_start_token_id, 
        initialize_from_vocab=True
    )
    model.set_input_embeddings(s_wte)
     
    #! fix model parameter
    parameters = list(model.parameters())
    for x in parameters[1:]:
        x.requires_grad = False
    