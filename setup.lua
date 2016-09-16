local weight_init = require 'weight-init'
local util = require 'util'

----------------------------------------------------------------------
-- parse command-line options
--
local opt = lapp[[
   -s,--save          (default "logs")          subdirectory to save logs
   -l,--load          (default "")              If nonempty, load previously saved model
   -o,--optimization  (default "SGD")           optimization: SGD | ADAM
   -e,--maxEpochs     (default 15)              Maximum number of epochs before halting
   -r,--learningRate  (default 0.1)             Learning rate
   -t,--traintype     (default "trans_only")    Use trans_only or regression (trans+rot)
   -b,--batchSize     (default 8)               Per-GPU batch size
   -u,--momentum      (default 0.9)             SGD momentum
   -w,--weightDecay   (default 1e-6)            L2 penalty on the weights
   -d,--dropout       (default 0)               Dropout fraction
   -n,--noiseSigma    (default 0)               Sigma for Gaussian noise added to training images
   -m,--model         (default "standardBlockModel") Model architecture to use
   --weight_init      (default "kaiming")       Initialization method to use
   -p,--noplot                                  plot while training
   --nogpu                                      Run without the GPU (runs with GPU by default)
   --numGPU           (default 0)               Number of GPUs to use if running on GPU (0: all available)
   --trainSeqNums     (default -1)              Train sequences to use. (-1 for all)
   --conover                                    Run controlled overfitting (10 examples, train/test) (defaults to false)
]]


opt.nBatchesPerSet = 500 -- number of minibatches to run between test intervals
opt.trainLogInterval = 100 -- Plot after this # minibatches

if opt.trainSeqNums == -1 then
   opt.trainSeqNums = {4,1,0,2,3,5,6,7}
else
   opt.trainSeqNums = {opt.trainSeqNums}
end

opt.interTestSeqNums = {9}
opt.testSeqNums = {8,9,10}

-- Only run for sequences that are present on the machine
opt.trainSeqNums = kitti.checkRequestedSequences(opt.trainSeqNums)
opt.testSeqNums = kitti.checkRequestedSequences(opt.testSeqNums)
opt.interTestSeqNums = kitti.checkRequestedSequences(opt.interTestSeqNums)


if opt.conover then
   torch.manualSeed(1)
   opt.noplot = true
   opt.weightDecay = 0
   opt.noiseSigma = 0
end

if opt.save == "logs" then -- change default
   opt.save = "/scratch/KITTINet/logs"
end

-- Define model to train or load pre-existing one
local modelDef
if opt.load == "" then
    if opt.traintype == "regression" then
        noutputs = 6
    elseif opt.traintype == "trans_only" then
      noutputs = 3
    end

    modelDef = architectures.createModel(kitti.imsize,noutputs,opt)
    modelDef = weight_init(modelDef,opt.weight_init)

else
    print("Loading model...")
    modelDef = util.loadDataParallel(opt.load,cutorch.getDeviceCount())
end

-- Set up optimization
local optConfig,optState
if opt.optimization == 'SGD' then
    optState = {
        learningRate = opt.learningRate,
        momentum = opt.momentum,
        learningRateDecay = 0
    }
   optConfig = optState
elseif opt.optimization == 'ADAM' then
    optState = {
        learningRate = opt.learningRate,
    }
    optConfig = optState
else
    error('unknown optimization method')
end

------------------------------------------------------------
-- Add GPU
local model
if opt.nogpu then
    print("WARNING: Not using GPU - will be very slow")
    model = modelDef
else
   if opt.conover then
      model = util.makeDataParallel(modelDef,1) -- single GPU
      opt.batchSize = 10 -- Force
   else
      local numGPU
      assert(opt.numGPU>=0,'numGPU must be nonnegative.')
      if opt.numGPU > 0 then
         numGPU = math.min(cutorch.getDeviceCount(),opt.numGPU)
         if opt.numGPU > numGPU then
            print(string.format('More GPUs requested than available; using all %d available.',numGPU))
         end
      else
         numGPU = cutorch.getDeviceCount()
      end
      model = util.makeDataParallel(modelDef,numGPU)
      opt.batchSize = cutorch.getDeviceCount()*opt.batchSize
   end

   collectgarbage()

   cutorch.setDevice(1) --  Default preferred GPU
   cudnn.convert(model,cudnn)
end

----------------------------------------------------------------------
-- loss function: negative log-likelihood
--
local criterion
if opt.traintype == "heatmap" then
    local azCriterion = nn.DistKLDivCriterion()
    azCriterion.sizeAverage = true
    local elCriterion = nn.DistKLDivCriterion()
    elCriterion.sizeAverage = true
    local rhoCriterion = nn.DistKLDivCriterion()
    rhoCriterion.sizeAverage = true
    criterion = nn.ParallelCriterion()
                    :add(azCriterion)
                    :add(elCriterion)
                    :add(rhoCriterion)
else
    criterion = nn.MSECriterion()
    criterion.sizeAverage = true
end

if not opt.nogpu then
    criterion:cuda()
end

return opt, model, criterion
