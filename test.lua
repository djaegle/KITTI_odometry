local util = require 'util'
require 'hdf5'

function outputEstimates(estimates,groundTruth,cameraType,adjacentIms,startIm,seqno,opt)
   -- Write out estimates and ground truth for a given sequence.
   local estimatesClean = torch.Tensor(estimates:size())
   local groundTruthClean = torch.Tensor(groundTruth:size())

   estimatesClean:copy(estimates)
   groundTruthClean:copy(groundTruth)
   estimatesClean = util.unnormalize(estimatesClean,opt)
   groundTruthClean = util.unnormalize(groundTruthClean,opt)

   -- Save to a python-readable format
   local recentEstimates = hdf5.open(string.format("%s/mostRecentEstimates_seq%02d.h5",opt.save,seqno),'w')
   recentEstimates:write('estimates',estimatesClean)
   recentEstimates:write('groundTruth',groundTruthClean)
   recentEstimates:write('rawPose',posegraph.posesSeqs[seqno+1]) -- compensate for lua indexing
   recentEstimates:write('cameraType',cameraType)
   recentEstimates:write('adjacentIms',adjacentIms)
   recentEstimates:write('startIm',startIm)
   recentEstimates:close()
end


function test(model, criterion, seqnums, maxtested, opt)
   -- Preallocate GPU data
   local inputs = torch.CudaTensor()
   local targets = torch.CudaTensor()

    maxtested = maxtested or math.huge -- cap on number tested
    local time = sys.clock()
    local err = torch.Tensor(#seqnums):fill(0)
    local percent_t_err = 0 -- For more intuitive reasoning
    local rot_m_err = 0
    local numErr = torch.Tensor(#seqnums):fill(0) -- Keep track of how many batches
    local outofbounds = torch.Tensor(#seqnums):fill(0)
    local numEdges = torch.Tensor(#seqnums):fill(0)
    local allEstimatesCPU = torch.Tensor()
    local allTargetsCPU = torch.Tensor()
    local allCameraType = torch.Tensor()
    local allAdjacentIms = torch.Tensor()
    local allStartIm = torch.Tensor()

    print('----------------------------------------------------------')
    for s=1,#seqnums do
        local seqno = seqnums[s]
        local imdata = kitti.loadImageSequence(seqno)
        local graph = kitti.loadSequenceGraph(seqno)
        numEdges[s] = math.min(maxtested, graph:size(1))

        local num_targets

        if opt.traintype == "heatmap" then
           error('Heatmap not configured correctly for testing')
        elseif opt.traintype == "trans_only" then
           num_targets = 3
        else
           -- Default to regression
           num_targets = 6
        end

        -- Put together record-keeping structures for this sequence
        allEstimatesCPU:resize(numEdges[s],6):fill(0)
        allTargetsCPU:resize(allEstimatesCPU:size()):fill(0)
        allCameraType:resize(numEdges[s]):fill(0)
        allAdjacentIms:resize(numEdges[s]):fill(0)
        allStartIm:resize(numEdges[s]):fill(0)



        print('<trainer> on testing Set:')
        for t = 1,numEdges[s],opt.batchSize do
            xlua.progress(t, numEdges[s])
            -- test samples
            -- create mini batch
            local last_entry = math.min(t+opt.batchSize-1,numEdges[s])
            local batch =
                    kitti.generateSequenceGraphBatch(
                            imdata,graph,t,last_entry,opt)

            local targetsCPU = 0
            local inputsCPU = batch.data
            if opt.traintype == "heatmap" then
                targetsCPU = {batch.azHeat,batch.elHeat,batch.rhoHeat}
            elseif opt.traintype == "trans_only" then
               targetsCPU = batch.translation
            elseif opt.traintype == "regression" then
               targetsCPU = batch.trans_rot
            else
               error('Unknown traintype.')
            end

            -- Copy batch data to GPU
            inputs:resize(inputsCPU:size()):copy(inputsCPU)
            targets:resize(targetsCPU:size()):copy(targetsCPU)
            local outputs = model:forward(inputs)

            -- Save local outputs/estimates here
            if not opt.noplot then
               local outputsCPU = torch.Tensor()
               outputsCPU:resize(outputs:size()):copy(outputs)

               allTargetsCPU[{{t,last_entry},{}}] = batch.trans_rot:type(allTargetsCPU:type())
               if opt.traintype == "trans_only" then
                  -- Use the true rotation as the estimate for reconstruction
                  allEstimatesCPU[{{t,last_entry},{1,3}}] = outputsCPU
                  allEstimatesCPU[{{t,last_entry},{4,6}}] = batch.trans_rot[{{},{4,6}}]:type(allEstimatesCPU:type())
               else
                  allEstimatesCPU[{{t,last_entry},{}}] = outputsCPU
               end

               -- And grab the relational information from the batch
               allCameraType[{{t,last_entry}}] = batch.cameraType
               allAdjacentIms[{{t,last_entry}}] = batch.adjacentIms
               allStartIm[{{t,last_entry}}] = batch.startIm

            end


            local e = criterion:forward(outputs, targets)

            if e ~= e or e < -math.huge or e > math.huge then -- check for nans
                outofbounds[s] = outofbounds[s] + 1
            else
                err[s] = err[s] + e
                numErr[s] = numErr[s] + 1
                if opt.traintype == "regression" then
                  local percent_t_err_i, rot_m_err_i = util.avgPercentError(outputs,targets,opt)
                  percent_t_err = percent_t_err_i + percent_t_err
                  rot_m_err = rot_m_err + rot_m_err_i
               elseif opt.traintype == "trans_only" then
                  local percent_t_err_i = util.avgPercentError(outputs,targets,opt)
                  percent_t_err = percent_t_err_i + percent_t_err
               else
                  error('Unknown traintype.')
               end
            end
        end

        -- Write out accumulated estimates/gt here
        if not opt.noplot then
           outputEstimates(allEstimatesCPU,allTargetsCPU,allCameraType,allAdjacentIms,allStartIm,seqno,opt)
        end

    end

    -- timing
    time = sys.clock() - time
    time = time / numEdges:sum()
    print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

    local percent_t_err_avg = percent_t_err/numErr:sum()
    testTPercentLogger:add{['Translation error (fraction) (test set)'] = percent_t_err_avg}

    local rot_m_err_avg = rot_m_err/numErr:sum()
    testRMeterLogger:add{['Rotation error (deg/m) (test set)'] = rot_m_err_avg}

    local testerr = err:sum()/numErr:sum()
    testLogger:add{['Raw mean error (test set)'] = testerr}

    -- print error
    print("==================")
    print(string.format("Total testing error: %.4f", testerr))
    print(string.format("Total out of bounds: %.4f", outofbounds:sum()/numErr:sum()))
    print("==================")

    return testerr
end
