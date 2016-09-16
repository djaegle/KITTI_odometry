require 'torch'
require 'image'
require 'paths'
local matio = require 'matio'

-- Custom modules
require 'pose-graph'


------------------------------------------------------
-- KITTI Dataset functions
------------------------------------------------------

kitti = {}

kitti.datasetDir = './KITTIData/torchData'
kitti.imageDir = './KITTIData/images'
kitti.statsLocation = paths.concat(posegraph.location,'stats.dat')
kitti.imsize = {93,307}
kitti.seqSizes = {4541, 1101, 4661,  801,  271, 2761, 1101, 1101, 4071, 1591, 1201}
local DENSITY = 14 -- This is our standard for now
local NCLASSES = (DENSITY+1)^2 - 3*DENSITY + 1 -- to get only unique ones
kitti.classes = {}
for i = 1,NCLASSES do
   kitti.classes[i] = tostring(i)
end
kitti.translations = matio.load(kitti.datasetDir .. "/translations.mat","T")
                        :type(torch.getdefaulttensortype())
kitti.sigma = 0.4;
kitti.gamma = 0.15;


local function dataLocation(seqno)
    return paths.concat(kitti.datasetDir, string.format('seq%02d.t7',seqno))
end

function kitti.checkRequestedSequences(seqnos)
   local seqnosChecked = {}
   for p,seqno in pairs(seqnos) do
      if paths.filep(dataLocation(seqno)) or paths.dirp(string.format("%s/%02d",kitti.imageDir,seqno)) then
         seqnosChecked[#seqnosChecked+1]=seqno
      else
         print(string.format('Could not find sequence # %d',seqno))
      end
   end

   assert(#seqnosChecked > 0,'No valid sequences specified in set.')

   return seqnosChecked
end

function kitti.createImageSequence(seqno)
    assert(0 <= seqno and seqno <= #kitti.seqSizes-1, 'Seqno is not in range 0 to 10')
    local dataloc = dataLocation(seqno) -- get dataset location
    if not paths.filep(dataloc) then
        local imdata = torch.Tensor(kitti.seqSizes[seqno+1],2,kitti.imsize[1],kitti.imsize[2])
        for i=1,kitti.seqSizes[seqno+1] do
            im = image.load(string.format('%s/%02d/%06d/img0.png',kitti.imageDir,seqno,i-1),1)
            imdata[{i,1,{},{}}] = im
            im = image.load(string.format('%s/%02d/%06d/img1.png',kitti.imageDir,seqno,i-1),1)
            imdata[{i,2,{},{}}] = im
        end
        local dataset = {}
        dataset.data = imdata
        torch.save(dataloc, dataset)
        print("Created Dataset")
    end
end

function kitti.loadImageSequence(seqno)
   kitti.createImageSequence(seqno)
   return kitti.loadImageSequenceData(seqno)
end

function kitti.loadImageSequenceData(seqno)
   return torch.load(dataLocation(seqno)).data:type(torch.getdefaulttensortype())
end

function kitti.loadSequenceGraph(seqno)
    posegraph.createPoseGraphs(seqno) -- Loads from disk if already made
   return matio.load(posegraph.graphLocation(seqno),"x"):type(torch.getdefaulttensortype())
end



kitti.rhoHist = torch.Tensor(
           {0, 0.3400, 0.6800, 1.0200, 1.3600, 1.7000, 2.0400, 2.3800, 2.7200, 3.0600, 3.4000,
            3.7400, 4.0800, 4.4200, 4.7600, 5.1000, 5.4400, 5.7800, 6.1200, 6.4600, 6.8000,
            7.1400, 7.4800, 7.8200, 8.1600, 8.5000, 8.8400, 9.1800, 9.5200, 9.8600 });
kitti.rhoSize = kitti.rhoHist:size(1)

kitti.azHist = torch.Tensor(
          {-3.1416, -2.9845, -2.8274, -2.6704, -2.5133, -2.3562, -2.1991, -2.0420, -1.8850, -1.7279,
           -1.5708, -1.4137, -1.2566, -1.0996, -0.9425, -0.7854, -0.6283, -0.4712, -0.3142, -0.1571,
            0, 0.1571, 0.3142, 0.4712, 0.6283, 0.7854, 0.9425, 1.0996, 1.2566, 1.4137, 1.5708,
            1.7279, 1.8850, 2.0420, 2.1991, 2.3562, 2.5133, 2.6704, 2.8274, 2.9845 });
kitti.azSize = kitti.azHist:size(1)

kitti.elHist = torch.Tensor(
          {-1.5708, -1.4923, -1.4137, -1.3352, -1.2566, -1.1781, -1.0996, -1.0210, -0.9425, -0.8639,
           -0.7854, -0.7069, -0.6283, -0.5498, -0.4712, -0.3927, -0.3142, -0.2356, -0.1571, -0.0785,
            0, 0.0785, 0.1571, 0.2356, 0.3142, 0.3927, 0.4712, 0.5498, 0.6283, 0.7069, 0.7854,
            0.8639, 0.9425, 1.0210, 1.0996, 1.1781, 1.2566, 1.3352, 1.4137, 1.4923 });
kitti.elSize = kitti.elHist:size(1)
kitti.sphericalHeatLength = kitti.rhoHist:size(1)+kitti.azHist:size(1)+kitti.elHist:size(1)

function l1normalize(x)
    return x:div(torch.sum(x))
end

function azHeatMapMaker(az)
    -- Implements:
    -- l1normalize(exp(-(min(abs(azhist-azz),min(abs(azhist-azz+2*pi),abs(azhist-azz-2*pi)))).^2./gamma.^2))
    local diffs = kitti.azHist-az
    local absdiffs = torch.abs(diffs)
    return l1normalize(absdiffs
                        :cmin(torch.abs(diffs+2*math.pi))
                        :cmin(torch.abs(diffs-2*math.pi))
                        :pow(2)
                        :div(-kitti.gamma^2)
                        :exp()
                )
end

function heatMapMaker(hist,x)
    -- Implements:
    local absdiffs = torch.abs(hist-x)
    return l1normalize(absdiffs
                        :pow(2)
                        :div(-kitti.gamma^2)
                        :exp()
                )
end

-- GRAPH:
-- ImIndex1, CamIndex1, ImIndex2, CamIndex2, TranslationBin, ...
--   TranslationX, TranslationY, TranslationZ, ExpRot1, ExpRot2, ExpRot3, ...
---  Azimuth, Elevation, Magnitude, Euler1, Euler2, Euler3
function kitti.generateSequenceGraphBatch(imdata,graph,start,finish,opt,graphInds)
   -- If graphInds is unspecified or set to 0, will return nPairs=(finish - start + 1) samples in deterministic order.
    -- If graphInds specified, will return nPairs random ones between the start and finish indices
    graphInds = graphInds or false

    if not graphInds then
        nPairs = finish-start+1
        graphInds = torch.range(start,finish)
    else
        nPairs = graphInds:size()[1]
    end

    local imagedata = torch.Tensor(nPairs,2,kitti.imsize[1],kitti.imsize[2])
    local translation = torch.Tensor(nPairs,3):fill(0)
    local rotation = torch.Tensor(nPairs,3):fill(0)
    local az = torch.Tensor(nPairs):fill(0)
    local el = torch.Tensor(nPairs):fill(0)
    local norms = torch.Tensor(nPairs):fill(0)
    local azHeat = torch.Tensor(nPairs,kitti.azHist:size(1))
    local elHeat = torch.Tensor(nPairs,kitti.elHist:size(1))
    local rhoHeat = torch.Tensor(nPairs,kitti.rhoHist:size(1))
    local sphericalHeat = torch.Tensor(nPairs, kitti.sphericalHeatLength):fill(0)
    local euler = torch.Tensor(nPairs,3):fill(0)
    local cameraType = torch.Tensor(nPairs):fill(0) -- 0 for 1->2, 1 for 1->1, 2 for 2->2, 3 for 2->1
    local adjacentIms = torch.Tensor(nPairs):fill(0)
    local startIm = torch.Tensor(nPairs):fill(0)

    -- TODO: Do euler angles heat map
    -- eulerHeat
    local labelsNums = torch.Tensor(nPairs):fill(0)
    for i = 1,nPairs do
        local graphindex = graphInds[i]
        local imindex1 = graph[graphindex][1]
        local imcamera1 = graph[graphindex][2]+1
        local imindex2 = graph[graphindex][3]
        local imcamera2 = graph[graphindex][4]+1
        if imindex1 > imdata:size(1) or imindex2 > imdata:size(1) then
            error("IMAGE INDEX ERROR")
        end
        local labelNum = graph[graphindex][5]
        if labelNum <= 0 or labelNum > #kitti.classes then
            error("CLASS INDEX ERROR")
        end
        labelsNums[i] = labelNum
        translation[i][1] = graph[graphindex][6]
        translation[i][2] = graph[graphindex][7]
        translation[i][3] = graph[graphindex][8]
        norms[i] = torch.norm(translation[i]) -- for now ignoring the entry in the table
        -- Generate heat map for spherical coordinates and Euler angles
        az[i] = graph[graphindex][12]
        el[i] = graph[graphindex][13]
        norms[i] = graph[graphindex][14]
        -- Generate the heat map
        azHeat[i] = azHeatMapMaker(az[i])
        elHeat[i] = heatMapMaker(kitti.elHist,el[i])
        rhoHeat[i] = heatMapMaker(kitti.rhoHist,norms[i])
        sphericalHeat[{i,{1,kitti.azHist:size(1)}}] = azHeat[i]
        sphericalHeat[{i,{1+kitti.azHist:size(1),kitti.elHist:size(1)+kitti.azHist:size(1)}}] =
           elHeat[i]
        sphericalHeat[{i,{1+kitti.elHist:size(1)+kitti.azHist:size(1),kitti.sphericalHeatLength}}] =
           rhoHeat[i]
        rotation[i][1] = graph[graphindex][9]
        rotation[i][2] = graph[graphindex][10]
        rotation[i][3] = graph[graphindex][11]
        euler[i][1] = graph[graphindex][15]
        euler[i][2] = graph[graphindex][16]
        euler[i][3] = graph[graphindex][17]

        adjacentIms[i] = imindex2==(imindex1+1) and 1 or 0
        startIm[i] = imindex1
        -- 0 for 1->2, 1 for 1->1, 2 for 2->2, 3 for 2->1
        if imcamera1 == 1 then
           if imcamera2 == 1 then
             cameraType[i] = 1
           else
             cameraType[i] = 0
           end
        else
           if imcamera2 == 1 then
             cameraType[i] = 3
           else
             cameraType[i] = 2
           end
        end

        -- Get images
        if opt.noiseSigma > 0 then
           im1 = imdata[{imindex1,imcamera1,{},{}}] + torch.randn(imdata[{imindex1,imcamera1,{},{}}]:size())*opt.noiseSigma
           im2 = imdata[{imindex2,imcamera2,{},{}}] + torch.randn(imdata[{imindex2,imcamera2,{},{}}]:size())*opt.noiseSigma
        else
           im1 = imdata[{imindex1,imcamera1,{},{}}]
           im2 = imdata[{imindex2,imcamera2,{},{}}]
        end
        imagedata[{i,1,{},{}}] = im1
        imagedata[{i,2,{},{}}] = im2
    end
    local dataset = {}
    dataset.data = imagedata

    -- Remove mean and norm mean of regression targets
    dataset.translation = (translation -
        torch.repeatTensor(kitti.transformStats.t_means,translation:size()[1],1):type(torch.getdefaulttensortype()))
        /kitti.transformStats.t_norm_mean
    dataset.euler = (euler -
    torch.repeatTensor(kitti.transformStats.eul_means,euler:size()[1],1):type(torch.getdefaulttensortype()))
        /kitti.transformStats.eul_norm_mean

    dataset.rotation = rotation
    dataset.az = az
    dataset.el = el
    dataset.norm = norm
    dataset.azHeat = azHeat
    dataset.elHeat = elHeat
    dataset.rhoHeat = rhoHeat
    dataset.sphericalHeat = sphericalHeat
    dataset.trans_rot = torch.cat(dataset.translation,dataset.euler,2)
    dataset.labels = labels
    dataset.labelsNums = labelsNums
    dataset.norms = norms
    dataset.cameraType = cameraType
    dataset.adjacentIms = adjacentIms
    dataset.startIm = startIm
    setmetatable(dataset, {__index = function(self, index)
                   return {self.data[index], self.translation[index], self.rotation[index]}
    end})
    return dataset
end



-- Get aggregate stats of the dataset
function kitti.getTransformStats(force)
    -- Recomputes (1) the mean of each translation dir and euler angle and (2)
    -- the mean norm of the T and R (euler angle) 3-vectors over the full graph.
    -- These values can be used to center and scale regression targets.
    force = force or false
    if paths.filep(kitti.statsLocation) and not force then
        return -- Don't remake it unnecessarily
    end

    local t_means = torch.Tensor(3):zero()
    local eul_means = torch.Tensor(3):zero()
    local t_norm_mean = 0
    local eul_norm_mean = 0
    local n_pts = 0
    for seqno = 0,#kitti.seqSizes-1 do
        local graph = kitti.loadSequenceGraph(seqno)
        n_pts = n_pts + kitti.seqSizes[seqno+1]
        t_means = t_means + graph[{{1,kitti.seqSizes[seqno+1]},{6,8}}]:sum(1) -- dims 6-8 are translation
        eul_means = eul_means + graph[{{1,kitti.seqSizes[seqno+1]},{15,17}}]:sum(1)

    end

    t_means = t_means / n_pts
    eul_means = eul_means / n_pts

    -- Do it again to get the norms
    for seqno = 0,#kitti.seqSizes-1 do
        local graph = kitti.loadSequenceGraph(seqno)
        t_norm_mean = t_norm_mean + torch.norm(graph[{{1,kitti.seqSizes[seqno+1]},{6,8}}]-
            torch.repeatTensor(t_means,kitti.seqSizes[seqno+1],1),2,2):sum() -- 2 norm along dim 2
        eul_norm_mean = eul_norm_mean + torch.norm(graph[{{1,kitti.seqSizes[seqno+1]},{15,17}}]-
            torch.repeatTensor(eul_means,kitti.seqSizes[seqno+1],1),2,2):sum() -- 2 norm along dim 2
    end
    t_norm_mean = t_norm_mean  / n_pts
    eul_norm_mean = eul_norm_mean / n_pts

    local statsTable = {}
    statsTable.t_means = t_means
    statsTable.eul_means = eul_means
    statsTable.t_norm_mean = t_norm_mean
    statsTable.eul_norm_mean = eul_norm_mean

    -- Save out t_means,eul_means,t_norm,eul_norm
    torch.save(kitti.statsLocation, statsTable)
end

function kitti.loadTransformStats()
    kitti.getTransformStats() -- Loads from disk if already made
    return torch.load(kitti.statsLocation)
end

kitti.transformStats = kitti.loadTransformStats()
