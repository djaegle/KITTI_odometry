-- Standard packages
require 'torch'
require 'nn'
require 'nngraph'
require 'cunn'
require 'cudnn'
require 'cutorch'
require 'nnx'

require 'custom-modules'

-- use floats, for SGD
torch.setdefaulttensortype('torch.FloatTensor')

local function shrink(size,n)
   return math.ceil(size/(2^n))
end


architectures = {}

function architectures.createModel(imsize,noutputs,opt)
   local model
   if opt.model == 'smallStandardBlockModel' then
      model = architectures.smallStandardBlockModel(imsize,noutputs,opt)
   elseif opt.model == 'standardBlockModel' then
      model = architectures.standardBlockModel(imsize,noutputs,opt)
   elseif opt.model == 'largeStandardBlockModel10' then
      model = architectures.largeStandardBlockModel10(imsize,noutputs,opt)
   elseif opt.model == 'largeStandardBlockModel' then
      model = architectures.largeStandardBlockModel(imsize,noutputs,opt)
   elseif opt.model == 'factorBlockOnly' then
      model = architectures.factorBlockOnly(imsize,noutputs,opt)
   elseif opt.model == 'resBuildingBlockOnly' then
      model = architectures.resBuildingBlockOnly(imsize,noutputs,opt)
   elseif opt.model == 'simpleResModel' then
      model = architectures.simpleResModel(imsize,noutputs,opt)
   elseif opt.model == 'resBuildingBlockModelEarly' then
      model = architectures.resBuildingBlockModelEarly(imsize,noutputs,opt)
   elseif opt.model == 'resBuildingBlockModelLate' then
      model = architectures.resBuildingBlockModelLate(imsize,noutputs,opt)
   elseif opt.model == 'resBuildingBlockModelLateFactor' then
      model = architectures.resBuildingBlockModelLateFactor(imsize,noutputs,opt)
   else
      error('Unknown model type.')
   end

   return model
end

-- Models using standard blocks

function architectures.smallStandardBlockModel(imsize,noutputs,opt)
    model = nn.Sequential()
    model:add(modules.StandardBlock(2, 128, 2, 3, 1))
    model:add(modules.StandardBlock(128, 512, 2, 3, 1))
    model:add(modules.StandardBlock(512, 1024, 2, 3, 1))
    model:add(modules.OutputBlock(1024,noutputs,shrink(imsize[1],3), shrink(imsize[2],3),opt,true))
    return model
end

function architectures.standardBlockModel(imsize,noutputs,opt)
    model = nn.Sequential()
    model:add(modules.StandardBlock(2, 64, 2, 3, 1))
    model:add(modules.StandardBlock(64, 128, 2, 3, 1))
    model:add(modules.StandardBlock(128, 256, 2, 3, 1))
    model:add(modules.StandardBlock(256, 512, 2, 3, 1))
    model:add(modules.StandardBlock(512, 1024, 2, 3, 1))
    model:add(modules.OutputBlock(1024,noutputs,shrink(imsize[1],5), shrink(imsize[2],5),opt,true))
    return model
end

function architectures.largeStandardBlockModel10(imsize,noutputs,opt)
    model = nn.Sequential()
    model:add(modules.StandardBlock(2, 64, 1, 3, 1))
    model:add(modules.StandardBlock(64, 64, 2, 3, 1))
    model:add(modules.StandardBlock(64, 128, 1, 3, 1))
    model:add(modules.StandardBlock(128, 128, 2, 3, 1))
    model:add(modules.StandardBlock(128, 256, 1, 3, 1))
    model:add(modules.StandardBlock(256, 256, 1, 3, 1))
    model:add(modules.StandardBlock(256, 256, 2, 3, 1))
    model:add(modules.StandardBlock(256, 512, 1, 3, 1))
    model:add(modules.StandardBlock(512, 512, 1, 3, 1))
    model:add(modules.StandardBlock(512, 512, 2, 3, 1))
    model:add(modules.OutputBlock(512,noutputs,shrink(imsize[1],4), shrink(imsize[2],4),opt,true))
    return model
end

function architectures.largeStandardBlockModel(imsize,noutputs,opt)
    model = nn.Sequential()
    model:add(modules.StandardBlock(2, 64, 1, 3, 1))
    model:add(modules.StandardBlock(64, 64, 2, 3, 1))
    model:add(modules.StandardBlock(64, 128, 1, 3, 1))
    model:add(modules.StandardBlock(128, 128, 2, 3, 1))
    model:add(modules.StandardBlock(128, 256, 1, 3, 1))
    model:add(modules.StandardBlock(256, 256, 1, 3, 1))
    model:add(modules.StandardBlock(256, 256, 2, 3, 1))
    model:add(modules.StandardBlock(256, 512, 1, 3, 1))
    model:add(modules.StandardBlock(512, 512, 1, 3, 1))
    model:add(modules.StandardBlock(512, 512, 2, 3, 1))
    model:add(modules.StandardBlock(512, 1024, 2, 3, 1))
    model:add(modules.OutputBlock(1024,noutputs,shrink(imsize[1],5), shrink(imsize[2],5),opt,true))
    return model
end

-- Models using factor blocks or residual layers

function architectures.factorBlockOnly(imsize,noutputs,opt)
   model = nn.Sequential()
   -- Begin Parallel Tracks
   model:add(nn.SplitTable(2))
        :add(nn.ParallelTable()
                :add(nn.Reshape(1,imsize[1],imsize[2],true))
                :add(nn.Reshape(1,imsize[1],imsize[2],true)))
        :add(modules.FactorBlock(1,2,imsize[1],imsize[2]))
        :add(nn.JoinTable(2))
        :add(modules.OutputBlock(2,noutputs,imsize[1],imsize[2],opt,true))
    return model
end

function architectures.resBuildingBlockOnly(imsize,noutputs,opt)
   model = nn.Sequential()
   -- Begin Parallel Tracks
   model:add(nn.SplitTable(2))
        :add(nn.ParallelTable()
                 :add(nn.Reshape(1,imsize[1],imsize[2],true))
                 :add(nn.Reshape(1,imsize[1],imsize[2],true)))
        :add(modules.ResBuildingBlockParallel(1,32,64,2,true))
        :add(nn.JoinTable(2))
        :add(modules.OutputBlock(128,noutputs,shrink(imsize[1],1),shrink(imsize[2],1),opt,true))
    return model
end


function architectures.simpleResModel(imsize,noutputs,opt)
    model = nn.Sequential()
    -- Begin Parallel Tracks
    model:add(nn.SplitTable(2))
         :add(nn.ParallelTable()
                 :add(nn.Reshape(1,imsize[1],imsize[2],true))
                 :add(nn.Reshape(1,imsize[1],imsize[2],true)))
    -- Initial Conv layer:
    model:add(modules.StandardBlockParallel(1, 64, 2, 7, 3))
    -- Bottleneck Blocks
    model:add(nn.JoinTable(2))
    model:add(modules.ResBuildingBlock(128,128,128,2))
    model:add(modules.StandardActivation(128)) -- Final activation after final ResBlock, as in [2]
    model:add(modules.OutputBlock(128,noutputs,shrink(imsize[1],2), shrink(imsize[2],2),opt,true))
    return model
end

function architectures.resBuildingBlockModelEarly(imsize,noutputs,opt)
    model = nn.Sequential()
    -- Initial Conv layer:
    model:add(modules.StandardBlock(2, 64, 2, 7, 3))
    -- Bottleneck Blocks
    model:add(modules.ResBuildingBlock(64,64,64,1,true))
    model:add(modules.ResBuildingBlock(64,64,64,1))
    model:add(modules.ResBuildingBlock(64,64,64,1))
    model:add(modules.ResBuildingBlock(64,128,128,2)) -- Stride here
    model:add(modules.ResBuildingBlock(128,128,128,1))
    model:add(modules.ResBuildingBlock(128,128,128,1))
    model:add(modules.ResBuildingBlock(128,128,128,1))
    model:add(modules.ResBuildingBlock(128,256,256,2)) -- Stride here
    model:add(modules.ResBuildingBlock(256,256,256,1))
    model:add(modules.ResBuildingBlock(256,256,256,1))
    model:add(modules.ResBuildingBlock(256,256,256,1))
    model:add(modules.ResBuildingBlock(256,256,256,1))
    model:add(modules.ResBuildingBlock(256,256,256,1))
    model:add(modules.ResBuildingBlock(256,512,512,2)) -- Stride here
    model:add(modules.ResBuildingBlock(512,512,512,1))
    model:add(modules.ResBuildingBlock(512,512,512,1))
    model:add(modules.StandardActivation(512)) -- Final activation after final ResBlock, as in [2]
    model:add(modules.OutputBlock(512,noutputs,shrink(imsize[1],4), shrink(imsize[2],4),opt,true))
    return model
end

function architectures.resBuildingBlockModelLate(imsize,noutputs,opt)
    model = nn.Sequential()
    -- Begin Parallel Tracks
    model:add(nn.SplitTable(2))
         :add(nn.ParallelTable()
                 :add(nn.Reshape(1,imsize[1],imsize[2],true))
                 :add(nn.Reshape(1,imsize[1],imsize[2],true)))
    -- Initial Conv layer:
    model:add(modules.StandardBlockParallel(1, 64, 2, 7, 3))
    -- Bottleneck Blocks
    model:add(modules.ResBuildingBlockParallel(64,64,64,1,true))
    model:add(modules.ResBuildingBlockParallel(64,64,64,1))
    model:add(modules.ResBuildingBlockParallel(64,64,64,1))
    model:add(modules.ResBuildingBlockParallel(64,128,128,2)) -- Stride here
    model:add(modules.ResBuildingBlockParallel(128,128,128,1))
    model:add(modules.ResBuildingBlockParallel(128,128,128,1))
    model:add(modules.ResBuildingBlockParallel(128,128,128,1))
    model:add(modules.ResBuildingBlockParallel(128,256,256,2)) -- Stride here
    model:add(modules.ResBuildingBlockParallel(256,256,256,1))
    model:add(modules.ResBuildingBlockParallel(256,256,256,1))
    model:add(modules.ResBuildingBlockParallel(256,256,256,1))
    model:add(modules.ResBuildingBlockParallel(256,256,256,1))
    model:add(modules.ResBuildingBlockParallel(256,256,256,1))
    model:add(nn.JoinTable(2))
    model:add(modules.ResBuildingBlock(512,512,512,2))
    model:add(modules.ResBuildingBlock(512,512,512,1))
    model:add(modules.ResBuildingBlock(512,512,512,1))
    model:add(modules.StandardActivation(512)) -- Final activation after final ResBlock, as in [2]
    model:add(modules.OutputBlock(512,noutputs,shrink(imsize[1],4), shrink(imsize[2],4),opt,true))
    return model
end

function architectures.resBuildingBlockModelLateFactor(imsize,noutputs,opt)
    model = nn.Sequential()
    -- Begin Parallel Tracks
    model:add(nn.SplitTable(2))
         :add(nn.ParallelTable()
                 :add(nn.Reshape(1,imsize[1],imsize[2],true))
                 :add(nn.Reshape(1,imsize[1],imsize[2],true)))
    -- Initial Conv layer:
    model:add(modules.StandardBlockParallel(1, 64, 2, 7, 3))
    -- Bottleneck Blocks
    model:add(modules.ResBuildingBlockParallel(64,64,64,1,true))
    model:add(modules.ResBuildingBlockParallel(64,64,64,1))
    model:add(modules.ResBuildingBlockParallel(64,64,64,1))
    model:add(modules.ResBuildingBlockParallel(64,128,128,2)) -- Stride here
    model:add(modules.ResBuildingBlockParallel(128,128,128,1))
    model:add(modules.ResBuildingBlockParallel(128,128,128,1))
    model:add(modules.ResBuildingBlockParallel(128,128,128,1))
    model:add(modules.ResBuildingBlockParallel(128,256,256,2)) -- Stride here
    model:add(modules.ResBuildingBlockParallel(256,256,256,1))
    model:add(modules.ResBuildingBlockParallel(256,256,256,1))
    model:add(modules.ResBuildingBlockParallel(256,256,256,1))
    model:add(modules.ResBuildingBlockParallel(256,256,256,1))
    model:add(modules.ResBuildingBlockParallel(256,256,256,1))
    model:add(modules.FactorBlock(256,256,shrink(imsize[1],3),shrink(imsize[2],3)))
    model:add(nn.JoinTable(2))
    model:add(modules.ResBuildingBlock(512,512,512,2))
    model:add(modules.ResBuildingBlock(512,512,512,1))
    model:add(modules.ResBuildingBlock(512,512,512,1))
    model:add(modules.StandardActivation(512)) -- Final activation after final ResBlock, as in [2]
    model:add(modules.OutputBlock(512,noutputs,shrink(imsize[1],4), shrink(imsize[2],4),opt,true))
    return model
end
