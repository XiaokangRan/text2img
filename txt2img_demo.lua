
require 'image'
require 'nn'
require 'nngraph'
require 'cunn'
require 'cutorch'
require 'cudnn'
require 'lfs'
torch.setdefaulttensortype('torch.FloatTensor')

local alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
local dict = {}
for i = 1,#alphabet do
    dict[alphabet:sub(i,i)] = i
end
ivocab = {}
for k,v in pairs(dict) do
  ivocab[v] = k
end

opt = {
  filenames = '',
  dataset = 'cub',
  batchSize = 16,        -- number of samples to produce
  noisetype = 'normal',  -- type of noise distribution (uniform / normal).
  imsize = 1,            -- used to produce larger images. 1 = 64px. 2 = 80px, 3 = 96px, ...
  noisemode = 'random',  -- random / line / linefull1d / linefull
  gpu = 1,               -- gpu mode. 0 = CPU, 1 = GPU
  display = 0,           -- Display image: 0 = false, 1 = true
  nz = 100,              
  doc_length = 201,
  queries = 'cub_queries.txt',
  checkpoint_dir = '',
  net_gen = '',
  net_txt = '',
  cont_codes = 5,         -- number of continuous latent codes
  cont_range = 2.0,       -- vary uniformly between [-cont_range, cont_range]
  cont_samples = 10,      -- number of images between [-cont_range, cont_range] to generate
}

for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end

noise = torch.Tensor(opt.batchSize, opt.nz, opt.imsize, opt.imsize)
net_gen = torch.load(opt.checkpoint_dir .. '/' .. opt.net_gen)
net_txt = torch.load(opt.net_txt)
if net_txt.protos ~=nil then net_txt = net_txt.protos.enc_doc end

net_gen:evaluate()
net_txt:evaluate()

-- Extract all text features.
local fea_txt = {}
-- Decode text for sanity check.
local raw_txt = {}
local raw_img = {}
for query_str in io.lines(opt.queries) do
  local txt = torch.zeros(1,opt.doc_length,#alphabet)
  for t = 1,opt.doc_length do
    local ch = query_str:sub(t,t)
    local ix = dict[ch]
    if ix ~= 0 and ix ~= nil then
      txt[{1,t,ix}] = 1
    end
  end
  raw_txt[#raw_txt+1] = query_str
  txt = txt:cuda()
  fea_txt[#fea_txt+1] = net_txt:forward(txt):clone()
end

if opt.gpu > 0 then
  require 'cunn'
  require 'cudnn'
  net_gen:cuda()
  net_txt:cuda()
  noise = noise:cuda()
end

local html = '<html><body><h1>Generated Images</h1><table border="1px solid gray" style="width=100%"><tr><td><b>Caption</b></td><td><b>Image</b></td></tr>'

local cont_range = torch.linspace(-opt.cont_range, opt.cont_range, opt.cont_samples)

for i = 1,#fea_txt do
  if opt.noisetype == 'uniform' then
    noise[{{}, {opt.cont_codes + 1, -1}, {}, {}}]:uniform(-1, 1)
  elseif opt.noisetype == 'normal' then
    noise[{{}, {opt.cont_codes + 1, -1}, {}, {}}]:normal(0, 1)
  end
  for j = 1,opt.cont_codes do
    for idx = 1,opt.cont_samples do
      k = cont_range[idx]
      print(string.format('generating %d of %d, code %d, value %f', i, #fea_txt, j, k))
      local cur_fea_txt = torch.repeatTensor(fea_txt[i], opt.batchSize, 1)
      local cur_raw_txt = raw_txt[i]
      noise[{{}, {1, opt.cont_codes}, {}, {}}]:zero()
      noise[{{}, j, {}, {}}]:fill(k)

      local images = net_gen:forward{noise, cur_fea_txt:cuda()}
      local visdir = string.format('results/%s', opt.dataset)
      lfs.mkdir('results')
      lfs.mkdir(visdir)
      local fname_png = string.format('%s/img_%d_%d_%d.png', visdir, i, j, idx)
      local fname_txt = string.format('%s/img_%d.txt', visdir, i)
      images:add(1):mul(0.5)
      --image.save(fname_png, image.toDisplayTensor(images,4,torch.floor(opt.batchSize/4)))
      image.save(fname_png, image.toDisplayTensor(images,4,opt.batchSize/2))
      html = html .. string.format('\n<tr><td>%s</td><td><img src="%s"></td></tr>',
                                   cur_raw_txt, fname_png)
      os.execute(string.format('echo "%s" > %s', cur_raw_txt, fname_txt))
    end
  end
end

html = html .. '</html>'
fname_html = string.format('%s.html', opt.dataset)
os.execute(string.format('echo "%s" > %s', html, fname_html))

