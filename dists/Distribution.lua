local dists = require 'dists.env'
local argcheck = require 'argcheck'
local class = require 'class'

-- Define the Distribution main class.
local Distribution = class.new('dists.Distribution')
dists.Distribution = Distribution

Distribution.__init = argcheck{
    {name = 'self', type = 'dists.Distribution'},
    call = function(self)
        error('Not Implemented')
    end
}

Distribution.dims = argcheck{
    {name = 'self', type = 'dists.Distribution'},
    call = function(self)
        error('Not Implemented')
    end
}

Distribution.dist_flat_dim = argcheck{
    {name = 'self', type = 'dists.Distribution'},
    call = function(self)
        error('Not Implemented')
    end
}

Distribution.prior_dist_info = argcheck{
    {name = 'self', type = 'dists.Distribution'},
    {name = 'batch_size', type = 'number'},
    call = function(self,batch_size)
        error('Not Implemented')
    end
}

-- function accepts a flat distribution and converts to the appropriate
-- paremters
Distribution.activate_dist = argcheck{
    {name = 'self', type = 'dists.Distribution'},
    {name = 'flat_dist', type = 'torch.DoubleTensor'},
    call = function(self,flat_dist)
        error('Not Implemented')
    end
}

Distribution.sample = argcheck{
    {name = 'self', type = 'dists.Distribution'},
    {name = 'params', type = 'table'},
    call = function(self,params)
        error('Not Implemented')
    end
}

Distribution.sample_prior = argcheck{
    {name = 'self', type = 'dists.Distribution'},
    {name = 'batch_size', type = 'number'},
    call = function(self,batch_size)
        return self:sample(self:prior_dist_info(batch_size))
    end
}

-- TODO: Revert back on *Tensor for x_var type -- its causing issues
Distribution.logli = argcheck{
    {name = 'self', type = 'dists.Distribution'},
    {name = 'x_var', type = 'torch.DoubleTensor'},
    {name = 'params', type = 'table'},
    call = function(self,x_var,params)
        error('Not Implemented')
    end
}

Distribution.logli_prior = argcheck{
    {name = 'self', type = 'dists.Distribution'},
    {name = 'x_var', type = 'torch.DoubleTensor'},
    call = function(self,x_var)
        return self:logli(x_var,self:prior_dist_info(x_var:size(1)))
    end
}

return Distribution
