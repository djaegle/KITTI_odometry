----------------------------------------------------------------------
-- Training egomotion estimation using the KITTI dataset
----------------------------------------------------------------------
-- Standard packages
require 'torch'
require 'nn'
require 'nngraph'
require 'cunn'
require 'cudnn'
require 'cutorch'
require 'nnx'
require 'optim'
require 'paths'
require 'math'

package.path = './utils/?.lua;' .. package.path

-- use floats, for SGD
torch.setdefaulttensortype('torch.FloatTensor')

-- Custom packages
require 'dataset-kitti'
require 'custom-modules'
require 'architectures'
require 'batch-generator'
local util = require 'util'

local opt, model, criterion = dofile('setup.lua')

paths.dofile('test.lua')
paths.dofile('train.lua')

-- Result logging
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
testTPercentLogger = optim.Logger(paths.concat(opt.save, 'testTPercent.log'))
testRMeterLogger = optim.Logger(paths.concat(opt.save, 'testRMeter.log'))
trainBatchLogger = optim.Logger(paths.concat(opt.save,'batchTrain.log')) -- for more detailed tracking
bestModelLogger = optim.Logger(paths.concat(opt.save,'bestModels.log')) -- to keep track of which models saved

-- Run training/testing
if not opt.conover then
    -- Initial test for baseline
    local bestTestErr = math.huge
    local nTest = 10000 -- Keep fixed to evaluate on the same subset

    model:evaluate()
    local currTestErr = test(model, criterion, opt.interTestSeqNums, nTest, opt)
    if not opt.noplot and opt.load then
        testLogger:style{['% mean error (test set)'] = '-'}
        testLogger:plot()
        if opt.traintype == "regression" then
            testTPercentLogger:style{['Translation error (fraction) (test set)'] = '-'}
            testTPercentLogger:plot()
            testRMeterLogger:style{['Rotation error (deg/m) (test set)'] = '-'}
            testRMeterLogger:plot()
        elseif opt.traintype == "trans_only" then
           testTPercentLogger:style{['Translation error (fraction) (test set)'] = '-'}
           testTPercentLogger:plot()
        end
    end
    local outerEpoch = 0
    while outerEpoch < opt.maxEpochs do
        local batchGenerator = BatchGenerator:new(opt)
        local innerEpoch = 0

        while batchGenerator:hasTrainingEdgesLeft() do
            -- train/test
            cutorch.synchronize()
            model:training()
            train(model, criterion, batchGenerator:generateTrainingBatches(), outerEpoch, innerEpoch, opt)

            cutorch.synchronize()
            model:evaluate()
            currTestErr = test(model, criterion, opt.interTestSeqNums, nTest, opt)
            if not opt.noplot then
                testLogger:style{['% mean error (test set)'] = '-'}
                testLogger:plot()
                if opt.traintype == "regression" then
                    testTPercentLogger:style{['Translation error (fraction) (test set)'] = '-'}
                    testTPercentLogger:plot()
                    testRMeterLogger:style{['Rotation error (deg/m) (test set)'] = '-'}
                    testRMeterLogger:plot()
                elseif opt.traintype == "trans_only" then
                    testTPercentLogger:style{['Translation error (fraction) (test set)'] = '-'}
                    testTPercentLogger:plot()
                end
            end
            -- Save model state if it produces the best test error
            if currTestErr < bestTestErr then
                bestTestErr = currTestErr
                bestModelLogger:add{['Current % mean error (test set)'] = bestTestErr,
                                    ['Outer epoch index'] = outerEpoch,['Inner epoch index'] = innerEpoch}
                if opt.save ~= "" then
                    util.saveNet(opt,paths,model,optConfig,optState,'best_test')
                end
            end
            innerEpoch = innerEpoch + 1
        end
        -- Save last epoch, regardless of how good it was
        if opt.save ~= "" and not conover then
            util.saveNet(opt,paths,model,optConfig,optState,'most_recent')
        end
        outerEpoch = outerEpoch + 1
    end
else
   local outerEpoch = 0
   print('Running controlled overfitting: single GPU, batch size 10, no dropout or weight decay:')
   while outerEpoch do
       -- train/test
       model:training()
       trainConover(model, criterion, opt.trainSeqNums[1], outerEpoch, opt)
       if not opt.noplot then
           trainLogger:style{['% mean class accuracy (train set)'] = '-'}
           trainLogger:plot()
       end
       outerEpoch = outerEpoch + 1
   end
end
