local dists = require 'dists.env'
local argcheck = require 'argcheck'
local class = require 'class'

local Product = class.new('dists.Product', 'dists.Distribution')
dists.Product = Product

-- child_dists - list of child distributions
Product.__init = argcheck{
    {name = 'self', type = 'dists.Product'},
    {name = 'child_dists', type = 'table'},
    call = function(self,child_dists)
        self.child_dists = child_dists
        return self
    end
}

Product.dims = argcheck{
    {name = 'self', type = 'dists.Product'},
    call = function(self)
        local dim = 0

        for i = 1, #self.child_dists do
            dim = dim + self.child_dists[i]:dims()
        end
        return dim
    end
}

Product.dist_flat_dim = argcheck{
    {name = 'self', type = 'dists.Product'},
    call = function(self)
        local dim = 0

        for i = 1, #self.child_dists do
            dim = dim + self.child_dists[i]:dist_flat_dim()
        end
        return dim
    end
}

Product.cum_flat_dims = argcheck{
    {name = 'self', type = 'dists.Product'},
    call = function(self)
        local retval = {}
        local cumsum = 0
        retval[1] = 0

        for i = 1, #self.child_dists do
            cumsum = cumsum + self.child_dists[i]:dist_flat_dim()
            retval[i+1] = cumsum
        end
        return retval
    end
}

-- Make this handle differently than tensorflow code where used
Product.prior_dist_info = argcheck{
    {name = 'self', type = 'dists.Product'},
    {name = 'batch_size', type = 'number'},
    call = function(self,batch_size)
        local retval = {}

        for i = 1, #self.child_dists do
            retval[i] = self.child_dists[i]:prior_dist_info(batch_size)
        end
        return retval
    end
}

Product.split_dist_flat = argcheck{
    {name = 'self', type = 'dists.Product'},
    {name = 'flat_dist', type = 'torch.DoubleTensor'},
    call = function(self,flat_dist)
        local cum_dims = self:cum_flat_dims()
        local retval = {}

        for i = 1, #self.child_dists do
            retval[i] = flat_dist[{{},{cum_dims[i]+1,cum_dims[i+1]}}]
        end
        return retval
    end
}

-- TODO: Hoping autograd will work fine here. Check
-- Make this handle differently than tensorflow code where used
Product.activate_dist = argcheck{
    {name = 'self', type = 'dists.Product'},
    {name = 'flat_dist', type = 'torch.DoubleTensor'},
    call = function(self,flat_dist)
        local retval = {}
        local split_flat_dist = self:split_dist_flat(flat_dist)

        for i = 1, #self.child_dists do
            retval[i] = self.child_dists[i]:activate_dist(split_flat_dist[i])
        end
        return retval
    end
}

-- sample function takes a table with n subtables for each dist params
Product.sample = argcheck{
    {name = 'self', type = 'dists.Product'},
    {name = 'params', type = 'table'},
    call = function(self,params)
        local retval = {}
        
        for i = 1, #self.child_dists do
            retval[i] = self.child_dists[i]:sample(params[i])
        end
        return torch.cat(retval,2)
    end
}

Product.sample_prior = argcheck{
    {name = 'self', type = 'dists.Product'},
    {name = 'batch_size', type = 'number'},
    call = function(self,batch_size)
        local retval = {}
        
        for i = 1, #self.child_dists do
            retval[i] = self.child_dists[i]:sample_prior(batch_size)
        end
        return torch.cat(retval,2)
    end
}

-- params is a table of subtables
Product.logli = argcheck{
    {name = 'self', type = 'dists.Product'},
    {name = 'x_var', type = 'torch.DoubleTensor'}, -- TODO: Recheck this...*Tensor causing errors
    {name = 'params', type = 'table'},
    call = function(self,x_var,params)
        local retval = torch.zeros(x_var:size(1),1)
        local cum_dims = self:cum_flat_dims()

        for i = 1, #self.child_dists do
            retval = retval + self.child_dists[i]:logli(x_var[{{},{cum_dims[i]+1,cum_dims[i+1]}}],params[i])
        end
        return retval
    end
}

return Product
