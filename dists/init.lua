local argcheck = require 'argcheck'

local dists = require 'dists.env'
require 'dists.Distribution'
require 'dists.Categorical'
require 'dists.Gaussian'
require 'dists.Product'
require 'dists.MutualInformationCriteria'

return dists
