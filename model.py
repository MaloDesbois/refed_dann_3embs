## modèle backbone ConvTran 

class tAPE(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0,dates=None):
        super(tAPE, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.dates=dates

        pe = torch.zeros(max_len, d_model)  # positional encoding

        if dates is None :
          position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        else:
          position=get_day_count(dates).unsqueeze(1)
        

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin((position * div_term)*(d_model/max_len))
        pe[:, 1::2] = torch.cos((position * div_term)*(d_model/max_len))
        pe = scale_factor * pe.unsqueeze(0)

        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """
        x = x + self.pe
        return self.dropout(x)


class Attention_Rel_Scl(nn.Module):
    def __init__(self, emb_size, num_heads, seq_len, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.scale = emb_size ** -0.5
        # self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)
        self.key = nn.Linear(emb_size, emb_size, bias=False)
        self.value = nn.Linear(emb_size, emb_size, bias=False)
        self.query = nn.Linear(emb_size, emb_size, bias=False)

        self.relative_bias_table = nn.Parameter(torch.zeros((2 * self.seq_len - 1), num_heads))
        coords = torch.meshgrid((torch.arange(1), torch.arange(self.seq_len)))
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords[1] += self.seq_len - 1
        relative_coords = rearrange(relative_coords, 'c h w -> h w c')
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.LayerNorm(emb_size)

    def forward(self, x, mask):
        batch_size, seq_len, _ = x.shape
        k = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # k,v,q shape = (batch_size, num_heads, seq_len, d_head)

        attn = torch.matmul(q, k) * self.scale
        if mask is not None :
            s=attn.shape

            mask=mask.unsqueeze(1)
            mask=mask.repeat(1,s[1],1)
            mask=mask.unsqueeze(2)
            mask=mask.repeat(1,1,s[2],1)
            attn=attn.masked_fill(mask==0,-10000)

        # attn shape (seq_len, seq_len)
        attn = nn.functional.softmax(attn, dim=-1)

        # Use "gather" for more efficiency on GPUs
        relative_bias = self.relative_bias_table.gather(0, self.relative_index.repeat(1, 8))
        relative_bias = rearrange(relative_bias, '(h w) c -> 1 c h w', h=1 * self.seq_len, w=1 * self.seq_len)
        attn = attn + relative_bias

        # distance_pd = pd.DataFrame(relative_bias[0,0,:,:].cpu().detach().numpy())
        # distance_pd.to_csv('scalar_position_distance.csv')

        out = torch.matmul(attn, v)
        # out.shape = (batch_size, num_heads, seq_len, d_head)
        out = out.transpose(1, 2)
        # out.shape == (batch_size, seq_len, num_heads, d_head)
        out = out.reshape(batch_size, seq_len, -1)
        # out.shape == (batch_size, seq_len, d_model)
        out = self.to_out(out)
        return out


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Permute(nn.Module):
    def forward(self, x):
        return x.permute(1, 0, 2)






class CasualConvTran(nn.Module):
    def __init__(self,role, config, num_classes,dates):
        super().__init__()
        # Parameters Initialization -----------------------------------------------
        channel_size, seq_len = config['Data_shape'][1], config['Data_shape'][2]
        emb_size = config['emb_size']
        num_heads = config['num_heads']
        dim_ff = config['dim_ff']
        self.dates=dates
        dropout=config['dropout']
        self.Fix_pos_encode = config['Fix_pos_encode']
        self.Rel_pos_encode = config['Rel_pos_encode']
        self.role = role
        if role == "dom" : 
            emb_size=emb_size*2
        elif role == "inv" :
            emb_size=emb_size
        else:
            print('role doit être dom ou inv')
        # Embedding Layer -----------------------------------------------------------
        self.Conv1 = nn.Sequential(nn.Conv1d(channel_size, emb_size, kernel_size=3,padding=1, stride=1),
                                          nn.BatchNorm1d(emb_size), nn.GELU())

        self.Conv2 = nn.Sequential(nn.Conv1d(emb_size, emb_size, kernel_size=3, stride=1,padding=1),
                                          nn.BatchNorm1d(emb_size), nn.GELU())

        self.causal_Conv3 = nn.Sequential(nn.Conv1d(emb_size, emb_size, kernel_size=3, stride=2, dilation=2),
                                          nn.BatchNorm1d(emb_size), nn.GELU())

        if self.Fix_pos_encode == 'tAPE':
            self.Fix_Position = tAPE(emb_size, dropout, seq_len,dates=dates)
        elif self.Fix_pos_encode == 'Sin':
            self.Fix_Position = tAPE(emb_size, dropout=config['dropout'], max_len=seq_len)
        elif self.Fix_pos_encode =='tAPE_sansDC':
          self.Fix_Position = tAPE_sansDC(emb_size,dropout=config['dropout'], max_len=seq_len)
        elif config['Fix_pos_encode'] == 'Learn':
            self.Fix_Position = LearnablePositionalEncoding(emb_size, dropout=config['dropout'], max_len=seq_len)

        if self.Rel_pos_encode == 'eRPE':
            self.attention_layer = Attention_Rel_Scl(emb_size, num_heads, seq_len, config['dropout'])
        elif self.Rel_pos_encode == 'Vector':
            self.attention_layer = Attention_Rel_Vec(emb_size, num_heads, seq_len, config['dropout'])
        else:
            self.attention_layer = Attention(emb_size, num_heads, config['dropout'])

        self.LayerNorm = nn.LayerNorm(emb_size, eps=1e-5)
        self.LayerNorm2 = nn.LayerNorm(emb_size, eps=1e-5)

        self.FeedForward = nn.Sequential(
            nn.Linear(emb_size, dim_ff),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(dim_ff, emb_size),
            nn.Dropout(config['dropout']))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        if role == 'dom':
            self.out1 = nn.Linear(emb_size//2, num_classes)
            self.out2 = nn.Linear(emb_size//2,num_classes)
        else :
            self.out = nn.Linear(emb_size,num_classes)
        self.emb_size=emb_size

    def forward(self, x,mask):
        #x = x.unsqueeze(1)
        x=torch.permute(x,(0,2,1))
        x_src = self.Conv1(x)
        x_src = self.Conv2(x_src)#.squeeze(2)
        x_src = x_src.permute(0, 2, 1)
        if self.Fix_pos_encode != 'None':
            x_src_pos = self.Fix_Position(x_src)
            att = x_src + self.attention_layer(x_src_pos,mask)
        else:
            att = x_src + self.attention_layer(x_src,mask)
        att = self.LayerNorm(att)
        out = att + self.FeedForward(att)
        out = self.LayerNorm2(out)
        out = out.permute(0, 2, 1)
        outa=self.flatten(out)
        out = self.gap(out)
        out = self.flatten(out)
        if self.role == "dom": 
            out_p2=out[:,:self.emb_size//2]
            
            out_inf = out[:,:self.emb_size//2]
            out_irr = out[:,self.emb_size//2:]
            
            out_inf = self.out1(out_inf)
            out_irr = self.out2(out_irr)
            out1_inf = outa[:,:outa.shape[1]//2]
            out1_irr = outa[:,outa.shape[1]//2:]
            return out_inf,out_irr,out1_inf,out1_irr,out_p2
        else :
            out_p1=out
            out=self.out(out)
            return out,outa,out_p1


## refed
class ConvTranDisentangle(torch.nn.Module):
    def __init__(self,config, num_classes=11,dates=None):
        super(ConvTranDisentangle, self).__init__()

        self.inv = CasualConvTran(role='inv',config=config,num_classes=num_classes,dates=dates)
        self.spec = CasualConvTran(role='dom',config=config,num_classes=8,dates=dates)
        self.dom_rev = nn.LazyLinear(8)
        self.classif = nn.LazyLinear(11)

    def forward(self, x,mask,alpha=1):
        classif_inv, inv_emb,classif_p1 = self.inv(x,mask)
        reverse_emb = ReverseLayerF.apply(inv_emb, alpha)
        adv_cl_dom = self.dom_rev(reverse_emb)
        classif_spec_inf,classif_spec_irr, spec_emb_inf,spec_emb_irr,classif_p2 = self.spec(x,mask)
        
        pred =  torch.concat((classif_p1,classif_p2),dim=1)
        
        pred = self.classif(pred)
        return classif_inv, inv_emb, spec_emb_inf,spec_emb_irr, classif_spec_inf,classif_spec_irr,adv_cl_dom,pred
