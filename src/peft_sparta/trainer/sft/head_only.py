class HeadOnly:
    def __init__(self, model, train_embeddings=False):
        
        self.model = model
        self.train_embeddings = train_embeddings # makes whole embeddings matrix trainable  
        
        head = 'score.weight' # classification head
        embed = model.get_input_embeddings().weight

        self.trainable_params = {}
        for name, param in model.named_parameters():
            if name == head or (param is embed and train_embeddings):
                self.trainable_params[name] = param    
            else:
                param.requires_grad_(False)

    def __repr__(self):
        return (f"HeadOnlyModel(train_embeddings={self.train_embeddings},"
                f"\n  {repr(self.model)}\n)")

    def named_parameters(self):
        return iter(self.trainable_params.items())

    def parameters(self):
        return iter(self.trainable_params.values())

    def num_parameters(self): 
        return sum(param.numel() for param in self.parameters())

    def print_trainable_parameters(self):
        n_trainable = self.num_parameters()
        pct = n_trainable/self.model.num_parameters()*100
        print(f"Num trainable parameters: {n_trainable:,d} ({pct:.5f}%)")

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)
