
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
    cont_codes = 3,         -- number of continuous latent codes
    disc_codes = 5,         -- number of discrete latent codes
    cont_range = 2.0,       -- vary uniformly between [-cont_range, cont_range]
    cont_samples = 10,      -- number of images between [-cont_range, cont_range] to generate (if not web)
    web = false,            -- use web frontend
    port = '8081'           -- which port to use
}

for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end

local latent_codes = opt.cont_codes + opt.disc_codes

noise = torch.Tensor(opt.batchSize, opt.nz, opt.imsize, opt.imsize)
net_gen = torch.load(opt.checkpoint_dir .. '/' .. opt.net_gen)
net_txt = torch.load(opt.net_txt)
if net_txt.protos ~=nil then net_txt = net_txt.protos.enc_doc end

net_gen:evaluate()
net_txt:evaluate()

if not opt.web then
    -- Extract all text features.
    fea_txt = {}
    -- Decode text for sanity check.
    raw_txt = {}
    raw_img = {}
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
end

if opt.gpu > 0 then
    require 'cunn'
    require 'cudnn'
    net_gen:cuda()
    net_txt:cuda()
    noise = noise:cuda()
end

if not opt.web then
    local html = '<html><body><h1>Generated Images</h1><table border="1px solid gray" style="width=100%"><tr><td><b>Caption</b></td><td><b>Image</b></td></tr>'

    local cont_range = torch.linspace(-opt.cont_range, opt.cont_range, opt.cont_samples)

    for i = 1,#fea_txt do
        if opt.noisetype == 'uniform' then
            noise[{{}, {latent_codes + 1, -1}, {}, {}}]:uniform(-1, 1)
        elseif opt.noisetype == 'normal' then
            noise[{{}, {latent_codes + 1, -1}, {}, {}}]:normal(0, 1)
        end
        for j = 1,latent_codes do
            local loop_limit = opt.cont_samples
            if j > opt.cont_codes then
                loop_limit = 1
            end
            for idx = 1,loop_limit do
                if j > opt.cont_codes then
                    k = 1
                else
                    k = cont_range[idx]
                end
                print(string.format('generating %d of %d, code %d, value %f', i, #fea_txt, j, k))
                local cur_fea_txt = torch.repeatTensor(fea_txt[i], opt.batchSize, 1)
                local cur_raw_txt = raw_txt[i]
                noise[{{}, {1, latent_codes}, {}, {}}]:zero()
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
end

if opt.web then
    function url_decode(str)
        str = string.gsub(str, "+", " ")
        str = string.gsub(str, "%%(%x%x)", function(h) return string.char(tonumber(h,16)) end)
        str = string.gsub(str, "\r\n", "\n")
        return str
    end

    local app = require('waffle')
    local htmlua = require('htmlua')
    app.set('public', '.')
    local js = [[
    update = function(id, value) {
        var xmlhttp = new XMLHttpRequest();

        xmlhttp.onreadystatechange = function() {
            if (xmlhttp.readyState == XMLHttpRequest.DONE ) {
                if (xmlhttp.status == 200) {
                    document.getElementById("main").src = document.getElementById("main").src + "?cache=" + (new Date()).getTime()
                }
            }
        };
        xmlhttp.open("GET", "update?c"+id+"="+value, true);
        xmlhttp.send();
    }

    ]]

    app.get('/', function(req, res)
        txt = torch.zeros(1,opt.doc_length,#alphabet)
        query_str = url_decode(req.url.args.text)

        for t = 1,opt.doc_length do
            local ch = query_str:sub(t,t)
            local ix = dict[ch]
            if ix ~= 0 and ix ~= nil then
                txt[{1,t,ix}] = 1
            end
        end
        txt = txt:cuda()
        fea_txt = net_txt:forward(txt):clone()

        if opt.noisetype == 'uniform' then
            noise[{{}, {latent_codes + 1, -1}, {}, {}}]:uniform(-1, 1)
        elseif opt.noisetype == 'normal' then
            noise[{{}, {latent_codes + 1, -1}, {}, {}}]:normal(0, 1)
        end
        cur_fea_txt = torch.repeatTensor(fea_txt, opt.batchSize, 1)
        cur_raw_txt = query_str
        noise[{{}, {1, latent_codes}, {}, {}}]:zero()

        images = net_gen:forward{noise, cur_fea_txt:cuda()}
        visdir = string.format('results/%s', opt.dataset)
        lfs.mkdir('results')
        lfs.mkdir(visdir)
        fname_png = string.format('%s/img_web.png', visdir)
        images:add(1):mul(0.5)
        image.save(fname_png, image.toDisplayTensor(images,4,opt.batchSize/2))

        local body_txt = { script { type='text/javascript', js },
            img { src=fname_png, id='main' }, br,
        }
        for i=1,opt.cont_codes do
            body_txt[#body_txt+1] = input { type='range', min=-opt.cont_range, max=opt.cont_range, value=0.0, step=0.1, onchange="update(" .. i .. ", this.value)" }
            body_txt[#body_txt+1] = br
        end
        if opt.disc_codes > 0 then
            local disc_range = torch.range(0,opt.disc_codes):totable()
            local select_options = each(disc_range,option)
            select_options.onchange = "update('disc',this.value)"
            body_txt[#body_txt+1] = htmlua.select(select_options)
        end
        body_txt[#body_txt+1] = br
        body_txt[#body_txt+1] = p { "Image is updated when slider is released. Don't update faster than the network can generate images!" }

        res.send(html { body(body_txt) })
    end)

    app.get('/update', function(req, res)
        local disc_activate = req.url.args['cdisc']
        if opt.disc_codes > 0 and disc_activate then
            for i = opt.cont_codes+1,latent_codes do
                req.url.args[string.format('c%d',i)] = 0
                if i==opt.cont_codes+disc_activate then
                    req.url.args[string.format('c%d',i)] = 1
                end
            end
        end
        for i = 1,latent_codes do
            if req.url.args[string.format('c%d', i)] ~= nil then
                noise[{{}, i, {}, {}}]:fill(req.url.args[string.format('c%d', i)])
            end
        end

        images = net_gen:forward{noise, cur_fea_txt:cuda()}
        images:add(1):mul(0.5)
        image.save(fname_png, image.toDisplayTensor(images,4,opt.batchSize/2))

        res.send('true')
    end)

    app.listen({port=opt.port})
end
