local util = require 'util' 

function train(model, criterion, batches, outerEpoch, innerEpoch, opt)
    local time = sys.clock()
    local parameters,gradParameters = model:getParameters()

    -- Preallocate GPU data
    local inputs = torch.CudaTensor()
    local targets = torch.CudaTensor()

    print('----------------------------------------------------------')
    print('<trainer> on training set:')
    print("<trainer> online epoch # " .. outerEpoch .. ', batch set ' .. innerEpoch .. ' [batchSize = ' .. opt.batchSize .. ']')

    local batchSize = opt.batchSize

    local err = 0
    local numEdges = batches.numEdges
    local t = 0
    local batchCounter = 0

    -- Train on batches drawn at random from the graph
    while batches:hasNextBatch() do
        batchCounter = batchCounter + 1 -- Keep track of overall batch for logging
        local batch = batches:getNextBatch()
        t = t + batch.numEdges
        xlua.progress(t, numEdges)

        local inputsCPU = batch.data
        local targetsCPU = 0
        if opt.traintype == "heatmap" then
            targetsCPU = {batch.azHeat,batch.elHeat,batch.rhoHeat}
        elseif opt.traintype == "trans_only" then
           targetsCPU = batch.translation
        else
           targetsCPU = batch.trans_rot
        end

        -- Copy batch data to GPU
        cutorch.synchronize()

        inputs:resize(inputsCPU:size()):copy(inputsCPU)
        targets:resize(targetsCPU:size()):copy(targetsCPU)

        -- create closure to evaluate f(X) and df/dX
        local feval = function(x)
            model:zeroGradParameters()
            -- evaluate function for complete mini batch
            local outputs = model:forward(inputs)
            local f = criterion:forward(outputs, targets)
            -- estimate df/dW
            local df_do = criterion:backward(outputs, targets)
            model:backward(inputs, df_do) -- Get the gradients
            -- return f and df/dX
            if opt.weightDecay ~= 0 then
                return f, (gradParameters:add(opt.weightDecay,x))
            else
                return f, gradParameters
            end
        end

        if opt.optimization == 'SGD' then
            -- Perform SGD step:
            _, fx = optim.sgd(feval, parameters, optConfig, optState)
            err = util.reportNAN(err,fx)
        elseif opt.optimization == 'ADAM' then
            -- Perform ADAM step:
            _, fx = optim.adam(feval, parameters, optConfig, optState)
            err = util.reportNAN(err,fx)
        else
            error('Unknown optimization method.')
        end
        if math.fmod(batchCounter-1,opt.trainLogInterval) == 0 then
           trainBatchLogger:add{['Raw error, batch mean (train set)'] = fx[1]}

           if not opt.noplot then
              trainBatchLogger:style{['Raw error, batch mean (train set)'] = '-'}
              trainBatchLogger:plot()
           end
        end
        cutorch.synchronize()
    end

    -- time taken
    time = sys.clock() - time
    print("")
    print("<trainer> total time "..(math.floor(time/60)).."m"..(time-60*math.floor(time/60)).."s. time to learn 1 sample = " .. ((time/numEdges)*1000) .. 'ms')

    local trainerr = err/(numEdges/batchSize)
    print("==================")
    print(string.format("Training error on batch set: %.4f",trainerr))
    print("==================")
    trainLogger:add{['% mean error (train set)'] = trainerr,
                        ['Outer epoch index'] = outerEpoch,['Inner epoch index'] = innerEpoch}

    cutorch.synchronize()
end

-- Controlled overfitting
function trainConover(model, criterion, seqno, epoch, opt)
    local time = sys.clock()
    local parameters,gradParameters = model:getParameters()

    -- Preallocate GPU data
    local inputs = torch.CudaTensor()
    local targets = torch.CudaTensor()

    local imdata = kitti.loadImageSequence(seqno)
    local graph = kitti.loadSequenceGraph(seqno)

    local err = 0

    -- Deterministic: just test on the first batch
    local numEdges = 10
    local batchSize = 1
    local batchCounter = 0

    for t = 1,numEdges,batchSize do
        batchCounter = batchCounter + 1 -- Keep track of overall batch for logging
        -- create mini batch
        local batch = {}
        batch =
             kitti.generateSequenceGraphBatch(
                     imdata,graph,1,numEdges,opt)

        local inputsCPU = batch.data
        local targetsCPU = 0
        if opt.traintype == "heatmap" then
            targetsCPU = {batch.azHeat,batch.elHeat,batch.rhoHeat}
        elseif opt.traintype == "trans_only" then
            targetsCPU = batch.translation
        else
           targetsCPU = batch.trans_rot
        end

        -- Copy batch data to GPU
        cutorch.synchronize()
        inputs:resize(inputsCPU:size()):copy(inputsCPU)
        targets:resize(targetsCPU:size()):copy(targetsCPU)

        -- create closure to evaluate f(X) and df/dX
        local feval = function(x)
            -- get new parameters
            if x ~= parameters then
               parameters:copy(x)
            end

            gradParameters:zero()

            -- evaluate function for complete mini batch
            local outputs = model:forward(inputs)
            local f = criterion:forward(outputs, targets)
            local df_do = criterion:backward(outputs, targets)
            model:backward(inputs, df_do)

            if opt.weightDecay ~= 0 then
                return f, (gradParameters:add(parameters:close():mul(opt.weightDecay)))
            else
                return f, gradParameters
            end
        end


        if opt.optimization == 'SGD' then
            _, fx = optim.sgd(feval, parameters, optConfig, optState)
            err = util.reportNAN(err, fx)

        elseif opt.optimization == 'ADAM' then
            _, fx = optim.adam(feval, parameters, optConfig, optState)
            err = util.reportNAN(err, fx)
        else
            error('Unknown optimization method.')
        end
        if math.fmod(batchCounter-1,opt.trainLogInterval) == 0 then
           trainBatchLogger:add{['Raw error, batch mean (train set)'] = fx[1]}
           if not opt.noplot then
              trainBatchLogger:style{['Raw error, batch mean (train set)'] = '-'}
              trainBatchLogger:plot()
           end
        end
        cutorch.synchronize()
    end

    -- time taken
    time = sys.clock() - time
    trainerr = err/(numEdges/batchSize)
    print(string.format("Training error on sequence %d: %e, iteration %d",seqno,trainerr,epoch))
    trainLogger:add{['Raw error, batch mean (train set)'] = trainerr}
    cutorch.synchronize()
end
