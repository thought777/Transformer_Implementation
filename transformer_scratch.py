# input embeddings defining
'''
    original sentence split into words seperated by space and then convert 
    to input ids (i.e. position in the vocabulary)
    converting it to embeddings [vector of size 512]
'''
import torch
import torch,math
import torch.nn as nn

class InputEmbeddings(nn.Module):
    
    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        #layer converting to embedding
        self.embedding=nn.Embedding(vocab_size,d_model)

    def forward(self,x):
        return self.embedding(x)*math.sqrt(self.d_model)
    
    #we also want to add positional embedding: which in turns tells us word position in the sentence
    #done using adding another vector of the same size

class PositionalEncoding(nn.Module):
    def __init__(self,d_model:int,seq_len:int,dropout:float) -> None:
        super().__init__()
        self.d_model=d_model
        self.seq_len=seq_len
        self.dropout=nn.Dropout(dropout)
        
        #seq_model -> d_model
        #creating a matrix of shape (seq_len,d_model)
        
        pe=torch.zeros(seq_len,d_model)
        # positional encoding : (pos,2i)=sin (pos/(10000)**d(2*i)/model)
        # positional encoding : (pos,2i+1)=cos (pos/(10000)**d(2*i)/model)
        #create a vector of shape (seq_length,1)
        position=torch.arange(0,seqlen,dtype=torch.float).unsqueeze(1) # (seq_len,1) #word inside the sentence
        
        #calculated in log term because log stability
        div_term=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        #Apply sin to even position
        pe[:,0::2]=torch.sin(position*div_term)
        #Apply cos to odd position
        pe[:,1::2]=torch.cos(position*div_term)
        
        
        #Add positional encoding to the input
        pe=pe.unsqueeze(0) #(1,seq_len,d_model)
        self.register_buffer('pe',pe) #tensor will be saved as model parameter
        
    def forward(self,x):
        x=x+(self.pe[:,:x.shape[1],:]).requires_grad_(False)
        return self.dropout(x)
    
#layer Normalization
class LayerNormalization(nn.Module):
    def __init__(self,eps:float=1e-6) -> None:
        super().__init__()
        self.eps=eps
        self.alpha=nn.Parameter(torch.ones(1)) # Multiplied   
        self.bias=nn.Parameter(torch.zeros(1)) # Added
        
        def forward(self,x):
            mean=x.mean(-1,keepdim=True)
            std=x.std(-1,keepdim=True)
            return self.alpha*(x-mean)/(std+self.eps)+self.bias

class FeedForwardBlock(nn.Module):
    def __init__(self,d_model:int,d_ff:int,dropout:float)->None:
        super().__init__()
        self.linear1=nn.Linear(d_model,d_ff) #w1 and b1
        self.dropout=nn.Dropout(dropout)
        self.linear2=nn.Linear(d_ff,d_model) #w2 and b2
        
        def forward(self,x):
            #x->linear1->relu->dropout->linear2
            #x->(batch_size,seq_len,d_model)->(batch_size,seq_len,d_ff)->(batch_size,seq_len,d_model)
            return self.linear2(self.dropout(torch.relu(self.linear1(x))))

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self,d_model:int,h:int,dropout:float) -> None:
        super().__init__()
        self.d_model=d_model
        self.h=h
        assert d_model%h==0, "d_model not divisible by h" #d_model must be divisible by h
        self.d_k=d_model//h #dimension of each head
        self.w_q=nn.Linear(d_model,d_model)#w_q
        self.w_k=nn.Linear(d_model,d_model) #w_k
        self.w_v=nn.Linear(d_model,d_model) #w_v
        
        self.w_o=nn.Linear(d_model,d_model) #w_o
        self.dropout=nn.Dropout(dropout) #
    @staticmethod
    def attention(query,key,value,mask,dropout:nn.Dropout):
        d_k=query.shape[-1]
        
        # (Batch,h,Seq_len,d_k) -> (Batch,h,Seq_len,Seq_len)
        attention_scores=(query @ key.transpose(-2,-1))/math.sqrt(d_k)
        #applying mask
        if mask is not None:
            attention_scores.masked_fill_(mask==0,-1e9)
        attention_scores=attention_scores.softmax(dim=-1) # (batch,h,seq_len,seq_len)
        if dropout is not None:
            attention_scores=dropout(attention_scores)            
        return (attention_scores@value),attention_scores #visulaizing purpose: returning attention scores
    def forward(self,q,k,v,mask):
        query=self.w_q(q) #(batch,Seq_len,d_model) --> (batch,Seq_len,d_model)
        key=self.w_k(k) #(batch,Seq_len,d_model) --> (batch,Seq_len,d_model)
        value=self.w_v(v) #(batch,Seq_len,d_model) --> (batch,Seq_len,d_model)
        
        # (Batch,Seq_len,d_model) -> (Batch,Seq_len,h,d_k) [by transpose] --> (Batch,h,Seq_len,d_k) :we want every head to see sequence_length and d_k
        query=query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2) 
        '''
            (B, seq_length, h, d_k)
            → transpose(1,2)
            → (B, h, seq_length, d_k)
        '''
        key=key.view(key.shape[0],key.shape[1],self.h,self.d_k).transpose(1,2)
        value=value.view(value.shape[0],value.shape[1],self.h,self.d_k).transpose(1,2)
        
        #Attention Calculation
        x,self.attention_scores=MultiHeadAttentionBlock.attention(query,key,value,mask,self.dropout)
        
        # (Batch,h,Seq_len,d_k) -> (Batch,Seq_len,h,d_k) [by transpose] --> (Batch,Seq_len,d_model)
        x=x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h*self.d_k) #d_model-> h*d_k
        return self.w_o(x)
    
class ResidualConnection(nn.Module):
    