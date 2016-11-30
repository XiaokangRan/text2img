local dists = require 'dists.env'
local argcheck = require 'argcheck'
local class = require 'class'

local Gaussian = class.new('dists.Gaussian', 'dists.Distribution')
dists.Gaussian = Gaussian

Gaussian.__init = argcheck{
    {name = 'self', type = 'dists.Gaussian'},
    {name = 'dim', type = 'number'},
    {name = 'fix_std', type = 'boolean', default = 'false'},
    call = function(self,dim,fix_std)
        self.dim = dim
        self.fix_std = fix_std
        return self
    end
}

Gaussian.dims = argcheck{
    {name = 'self', type = 'dists.Gaussian'},
    call = function(self)
        return self.dim
    end
}

Gaussian.dist_flat_dim = argcheck{
    {name = 'self', type = 'dists.Gaussian'},
    call = function(self)
        return 2*self.dim
    end
}

Gaussian.prior_dist_info = argcheck{
    {name = 'self', type = 'dists.Gaussian'},
    {name = 'batch_size', type = 'number'},
    call = function(self,batch_size)
        local mean = torch.zeros(batch_size,self.dim)
        local stddev = torch.ones(batch_size,self.dim)
        return {mean=mean,stddev=stddev}
    end
}

-- TODO: Hoping autograd will work fine here. Check
Gaussian.activate_dist = argcheck{
    {name = 'self', type = 'dists.Gaussian'},
    {name = 'flat_dist', type = 'torch.DoubleTensor'},
    call = function(self,flat_dist)
        local mean = flat_dist[{{},{1,self.dim}}]
        local stddev
        if self.fix_std then
            stddev = torch.ones(mean:size())
        else
            stddev = torch.sqrt(torch.exp(flat_dist[{{},{self.dim+1,2*self.dim}}]))
        end
        return {mean=mean,stddev=stddev}
    end
}

-- sample function takes a table with keys mean,stddev which are batch_size x dim tensors
Gaussian.sample = argcheck{
    {name = 'self', type = 'dists.Gaussian'},
    {name = 'params', type = 'table'},
    call = function(self,params)
        local mean = params.mean
        local stddev = params.stddev
        local epsilon = torch.randn(mean:size())
        return mean + torch.cmul(stddev,epsilon)
    end
}

-- logli function calculates the log likelihood given the pdf in params and
-- observed variable.
-- params - table with keys mean,stddev which are batch size x dim tensors of
-- observation probabilities
-- x_var - batch size x dim tensor of actual points
Gaussian.logli = argcheck{
    {name = 'self', type = 'dists.Gaussian'},
    {name = 'x_var', type = 'torch.DoubleTensor'}, -- TODO: Recheck this...*Tensor causing errors
    {name = 'params', type = 'table'},
    call = function(self,x_var,params)
        local mean = params.mean
        local stddev = params.stddev
        local epsilon = torch.cdiv(x_var-mean,stddev+dists.tiny)
        local unreduced_result = -0.5*torch.log(2*math.pi) - torch.log(stddev + dists.tiny) - 0.5*torch.pow(epsilon,2)
        return torch.sum(unreduced_result,2)
    end
}

return Gaussian
