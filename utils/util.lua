require 'cunn'

local util = {}

------------------------------
-- Simple convenience functions
------------------------------
function util.reportNAN(err, fx)
    if (fx[1] ~= fx[1]) then
        print(err)
        print(t)
        error("ERROR ABORT - NAN FOUND")
    end
    return err + fx[1]
end

function util.deepcopy(orig)
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
        copy = {}
        for orig_key, orig_value in next, orig, nil do
            copy[util.deepcopy(orig_key)] = util.deepcopy(orig_value)
        end
        setmetatable(copy, util.deepcopy(getmetatable(orig)))
    else -- number, string, boolean, etc
        copy = orig
    end
    return copy
end

---------------------------
-- Network saving utilities
---------------------------

function util.makeDataParallel(model, nGPU)
   print('converting module to nn.DataParallelTable')
   assert(nGPU <= cutorch.getDeviceCount(), 'number of GPUs less than nGPU specified')
   local model_single = model
   model = nn.DataParallelTable(1)
   for i=1, nGPU do
      cutorch.setDevice(i)
      model:add(model_single:clone():cuda(), i)
   end
   cutorch.setDevice(1)

   return model
end

function util.saveDataParallel(filename, model)
   if torch.type(model) == 'nn.DataParallelTable' then
      torch.save(filename, model:get(1))
   elseif torch.type(model) == 'nn.Sequential' then
      local temp_model = nn.Sequential()
      for i, module in ipairs(model.modules) do
         if torch.type(module) == 'nn.DataParallelTable' then
            temp_model:add(module:get(1)) -- strip DPT
         else
            temp_model:add(module)
         end
      end
      torch.save(filename, temp_model:float())
   else
      error('This saving function only works with Sequential or DataParallelTable modules.')
   end
end

function util.loadDataParallel(filename, nGPU)
   local model = torch.load(filename)
   if torch.type(model) == 'nn.DataParallelTable' then
      return makeDataParallel(model:get(1):float(), nGPU)
   elseif torch.type(model) == 'nn.Sequential' then
      for i,module in ipairs(model.modules) do
         if torch.type(module) == 'nn.DataParallelTable' then
            model.modules[i] = makeDataParallel(module:get(1):float(), nGPU)
         end
      end
      return model
   else
      error('The loaded model is not a Sequential or DataParallelTable module.')
   end
end

function util.saveNet(opt,paths,model,optConfig,optState,model_id)
   -- Save out the current model state.
   local filename = paths.concat(opt.save, model_id .. '.net')
   local optFilename = paths.concat(opt.save, model_id .. '_opt.t7')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   local epoch_written = paths.filep(filename)

   print('Saving network to '..filename)
   -- Save out the copy from ,a single GPU: don't need the 4x rep
   collectgarbage()
   model:clearState() -- clear intermediary states before saving.
   util.saveDataParallel(filename,model)
   -- Save optimization state as well
   print('Saving optimization state to '..optFilename)
   local optData = {['optState'] = optState, ['optConfig'] = optConfig}
   torch.save(optFilename, optData)
end



------------------------------
-- Data manipulation utilities
------------------------------

function util.unnormalize(estimates,opt)
   -- Returns an unnormalized tensor of heading values. Translation and rotation modes
   -- supported.
   local N = estimates:size(1)
   local estimatesOut = estimates:clone()

   estimatesOut[{{},{1,3}}] = estimates[{{},{1,3}}]*kitti.transformStats.t_norm_mean +
   torch.repeatTensor(kitti.transformStats.t_means,N,1):type(estimates:type())

   if opt.traintype == "regression" or estimates:size(2) == 6 then
      estimatesOut[{{},{4,6}}] = estimates[{{},{4,6}}]*kitti.transformStats.eul_norm_mean +
      torch.repeatTensor(kitti.transformStats.eul_means,N,1):type(estimates:type())
   end

   return estimatesOut
end


function util.eul2Rot(euls)
   --  Converts the XYZ intrinsic Euler angle representation of a rotation to a rotation matrix.
   --  Each column of euls is [psi,theta,phi].T, where:
   --  psi: rotation about x
   --  theta: rotation about y
   --  phi: rotation about z

   local psi = euls[{{},{1}}]:squeeze()
   local theta = euls[{{},{2}}]:squeeze()
   local phi = euls[{{},{3}}]:squeeze()
   local rot_mat = torch.Tensor(3,3,theta:size(1)):type(euls:type())

   local ctheta = torch.cos(theta)
   local stheta = torch.sin(theta)
   local cpsi = torch.cos(psi)
   local spsi = torch.sin(psi)
   local cphi = torch.cos(phi)
   local sphi = torch.sin(phi)

   rot_mat[{{1},{1},{}}] = torch.cmul(ctheta,cphi)
   rot_mat[{{1},{2},{}}] = torch.cmul(torch.cmul(spsi,stheta),cphi) - torch.cmul(cpsi,sphi)
   rot_mat[{{1},{3},{}}] = torch.cmul(torch.cmul(cpsi,stheta),cphi) + torch.cmul(spsi,sphi)
   rot_mat[{{2},{1},{}}] = torch.cmul(ctheta,sphi)
   rot_mat[{{2},{2},{}}] = torch.cmul(torch.cmul(spsi,stheta),sphi) + torch.cmul(cpsi,cphi)
   rot_mat[{{2},{3},{}}] = torch.cmul(torch.cmul(cpsi,stheta),sphi) - torch.cmul(spsi,cphi)
   rot_mat[{{3},{1},{}}] = -stheta
   rot_mat[{{3},{2},{}}] = torch.cmul(spsi,ctheta)
   rot_mat[{{3},{3},{}}] = torch.cmul(cpsi,ctheta)

    return rot_mat
end

function util.getRelativeAngles(trueRots, estRots)
   -- Returns the relative angle (in degrees) between two sets of rotation matrices.
   -- Both trueRots and estRots are of size 3x3xN and contain N rotation matrices.

   local trueRotMat = util.eul2Rot(trueRots)
   local estRotMat = util.eul2Rot(estRots)

   local relAngles = torch.Tensor(trueRotMat:size(3)):type(trueRotMat:type())
   local tmpMat = torch.Tensor(3,3):type(trueRotMat:type())
   for i = 1,relAngles:size(1) do
      tmpMat:fill(0) -- reset
      tmpMat:addmm(estRotMat[{{},{},{i}}]:squeeze():transpose(1,2),trueRotMat[{{},{},{i}}]:squeeze())

      -- Find the cosine of the relative rotation
      relAngles[i] = (torch.trace(tmpMat) - 1) / 2
   end

   --To account for numerical instability, round cosine to -1 or 1
   relAngles:cmin(relAngles,torch.Tensor(relAngles:size()):type(relAngles:type()):fill(1))
   relAngles:cmax(relAngles,torch.Tensor(relAngles:size()):type(relAngles:type()):fill(-1))
   relAngles:acos()

   return relAngles*180/math.pi
end

function util.avgPercentError(outputs,targets,opt)
    N = outputs:size(1)

    -- Translation error in percent
    local targetNorm = util.unnormalize(targets,opt)
    local outputNorm = util.unnormalize(outputs,opt)
    local metersTravelled = torch.norm(targetNorm[{{},{1,3}}],2,2)
    local errTrans = torch.norm(outputNorm[{{},{1,3}}]-targetNorm[{{},{1,3}}],2,2):cdiv(metersTravelled):mean()

    -- Rotation error in degrees / meter
    local errRot = 0
    if opt.traintype == "regression" then
      local relAngle = util.getRelativeAngles(targetNorm[{{},{4,6}}],outputNorm[{{},{4,6}}])
      errRot = relAngle:cdiv(metersTravelled):mean()
    end

    return errTrans, errRot
end

return util
