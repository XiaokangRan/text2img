local dists = require 'dists.env'
local argcheck = require 'argcheck'
local class = require 'class'
local grad = require 'autograd'

local Categorical = class.new('dists.Categorical', 'dists.Distribution')
dists.Categorical = Categorical

Categorical.__init = argcheck{
    {name = 'self', type = 'dists.Categorical'},
    {name = 'dim', type = 'number'},
    call = function(self,dim)
        self.dim = dim
        return self
    end
}

Categorical.dims = argcheck{
    {name = 'self', type = 'dists.Categorical'},
    call = function(self)
        return self.dim
    end
}

Categorical.dist_flat_dim = argcheck{
    {name = 'self', type = 'dists.Categorical'},
    call = function(self)
        return self.dim
    end
}

Categorical.prior_dist_info = argcheck{
    {name = 'self', type = 'dists.Categorical'},
    {name = 'batch_size', type = 'number'},
    call = function(self,batch_size)
        local prob = torch.ones(batch_size,self.dim)/self.dim
        return {prob=prob}
    end
}

-- TODO: Hoping autograd will work fine here. Check
Categorical.activate_dist = argcheck{
    {name = 'self', type = 'dists.Categorical'},
    {name = 'flat_dist', type = 'torch.DoubleTensor'},
    call = function(self,flat_dist)
        return {prob=grad.nn.SoftMax()(flat_dist)}
    end
}

-- sample function takes a table with key prob which is a batch_size x dim tensor
-- keep the prob matrix as a probability distibution, it will work with any
-- positive values however treating them as weights.
Categorical.sample = argcheck{
    {name = 'self', type = 'dists.Categorical'},
    {name = 'params', type = 'table'},
    call = function(self,params)
        local prob = params.prob
        local ids = torch.multinomial(prob,1,true)
        local onehot = torch.eye(self.dim)
        return onehot:gather(1,ids:repeatTensor(1,self.dim))
    end
}

-- logli function calculates the log likelihood given the pdf in params and
-- observed variable.
-- params - table with key prob which is a batch size x dim tensor of
-- observation probabilities
-- x_var - batch size x dim tensor of actual points
Categorical.logli = argcheck{
    {name = 'self', type = 'dists.Categorical'},
    {name = 'x_var', type = 'torch.DoubleTensor'}, -- TODO: Recheck this...*Tensor causing errors
    {name = 'params', type = 'table'},
    call = function(self,x_var,params)
        local prob = params.prob
        return torch.sum(torch.cmul(torch.log(prob+dists.tiny),x_var),2)
    end
}

return Categorical
