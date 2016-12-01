local dists = require 'dists.env'
local argcheck = require 'argcheck'
local class = require 'class'
local nn = require 'nn'
local grad = require 'autograd'

local MutualInformationCriteria,Parent = torch.class('dists.MutualInformationCriteria', 'nn.Criterion', dists)

MutualInformationCriteria.infogan_loss_fn = argcheck{
    {name = 'dist', type = 'dists.Distribution'},
    call = function(dist)
        -- In our case 1st input needs to be only a flat vector but wrapping it
        -- with a table for autograd. Check if needed.
        return function(pdf,target)
            -- Q output
            local out_params = dist:activate_dist(pdf.input)
            -- Q(c|x)
            local cross_entropy = torch.mean(-dist:logli(target,out_params))
            -- Q(c)
            local entropy = torch.mean(-dist:logli_prior(target))

            local mi_est = entropy - cross_entropy
            -- need to maximize this so minimize negative
            return -mi_est
        end
    end
}

MutualInformationCriteria.__init = argcheck{
    {name = 'self', type = 'dists.MutualInformationCriteria'},
    {name = 'dist', type = 'dists.Distribution'},
    call = function(self,dist)
        Parent.__init(self)
        self.loss = grad(self.infogan_loss_fn(dist))
    end
}

MutualInformationCriteria.updateOutput = argcheck{
    {name = 'self', type = 'dists.MutualInformationCriteria'},
    {name = 'input', type = 'torch.DoubleTensor'},
    {name = 'target', type = 'torch.DoubleTensor'},
    call = function(self,input,target)
        local grads,loss = self.loss({input=input},target)
        self.output = loss
        self.gradInput = grads.input

        return self.output
    end
}

MutualInformationCriteria.updateGradInput = argcheck{
    {name = 'self', type = 'dists.MutualInformationCriteria'},
    {name = 'input', type = 'torch.DoubleTensor'},
    {name = 'target', type = 'torch.DoubleTensor'},
    call = function(self,input,target)
        -- gradInput updated during updateOutput so just return
        return self.gradInput
    end
}

return MutualInformationCriteria
