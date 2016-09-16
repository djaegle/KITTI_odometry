require 'math'
require 'torch'
require 'image'
require 'paths'

local debugger = require('fb.debugger')

-- My modules
local util = require('util')

----------------------------------
-- BatchGenerator Class
---------------------------------
BatchGenerator = {
    trainSeqNums = {4,1,0,2,3,5,6,7}, -- Sequence numbers we are testing on
    interTestSeqNums = {9}, -- Sequence numbers we are testing on
    testSeqNums = {8,9,10}, -- Sequence numbers we are testing on
    trainEdgeSet = {}, -- Keeps tracks of edges (image pairs in pose graph) we haven't used
    interEdgeSet = {}, --
    testEdgeSet = {}, --
    traintype = "", -- Heatmap or regression?
    batchSize = 32, -- How many image pairs per batch?
    nBatchesPerSet = 200, -- How many batches per runthrough?
}

function BatchGenerator:initialize(opt)
    -- Gather necessary data
    print('Initializing batch generator...')
    self.graphs = {}
    self.imdata = {}

    self.trainSeqNums = opt.trainSeqNums
    self.interTestSeqNums = opt.interTestSeqNums
    self.testSeqNums = opt.testSeqNums

    for p,v in pairs(self.trainSeqNums) do
        self.imdata[v] = kitti.loadImageSequence(v)
        self.graphs[v] = kitti.loadSequenceGraph(v)
        collectgarbage()
    end

    -- Create Edge sets
    self.trainEdgeSet = self:createEdgeSet(self.trainSeqNums)

    self.batchSize = opt.batchSize
    self.nBatchesPerSet = opt.nBatchesPerSet
    -- Currently unused
    -- self.interEdgeSet = self:createEdgeSet(self.interTestSeqNums)
    -- self.testEdgeSet = self:createEdgeSet(self.testSeqNums)

    if opt.traintype == "regression" then
        self.traintype = "regression"
    elseif opt.traintype == "trans_only" then
        self.traintype = "trans_only"
    elseif opt.traintype == "heatmap" then
        self.traintype = "heatmap"
    else
        error("BatchGenerator: Unsupported Train Type: " .. opt.traintype)
    end
end

function BatchGenerator:new(opt)
    local obj = util.deepcopy(BatchGenerator)
    obj:initialize(opt)
    return obj
end

-- Creates set of edges that we haven't checked yet
function BatchGenerator:createEdgeSet(seqnos)
    local edgeSet = {}

    for p,seqno in pairs(seqnos) do
        edgeSet[seqno] = {}
        local size = self.graphs[seqno]:size(1)
        local edgeSetSeqno = {}
        for j=1,size do
            edgeSetSeqno[j] = {seqno,j}
        end
        edgeSet[seqno] = {edgeSetSeqno, size}
    end
    return edgeSet
end

function BatchGenerator:hasTrainingEdgesLeft()
    for i,v in pairs(self.trainEdgeSet) do
        if (self.trainEdgeSet[i][2] > 0) then
            return true
        end
    end
    return false
end

function BatchGenerator:hasInterTestingEdgesLeft()
    for p,i in pairs(self.interEdgeSet) do
        if (self.interEdgeSet[i][2] > 0) then
            return true
        end
    end
    return false
end

function BatchGenerator:hasTestingEdgesLeft()
    for p,i in pairs(self.testEdgeSet) do
        if (self.testEdgeSet[i][2] > 0) then
            return true
        end
    end
    return false
end

----------------------------------------
-- Batch class
----------------------------------------

Batch = {
    traintype = {},
    data = {},
    edges = {},
    trans_rot = {},
    translation = {},
    azHeat = {},
    elHeat = {},
    rhoHeat = {},
    batchSize = BatchGenerator.batchSize, -- How many image pairs per batch?
    nBatchesPerSet = BatchGenerator.nBatchesPerSet, -- How many batches per runthrough?
    imsigma = 0.02,
    index = 1,
};

-- Batch functions
function Batch:initTestData(numEdges)
    if self.traintype == "regression" then
        self.trans_rot = torch.zeros(numEdges,6)
    elseif self.traintype == "trans_only" then
        self.translation = torch.zeros(numEdges,3)
    else
        self.azHeat = torch.Tensor(numEdges,kitti.azHist:size(1))
        self.elHeat = torch.Tensor(numEdges,kitti.elHist:size(1))
        self.rhoHeat = torch.Tensor(numEdges,kitti.rhoHist:size(1))
    end
end

function Batch:getTestData(edge,i)
        -- Grab training type data
        if self.traintype == "regression" then
            self.trans_rot[i][1] = (edge[6] - kitti.transformStats.t_means[1])/kitti.transformStats.t_norm_mean
            self.trans_rot[i][2] = (edge[7] - kitti.transformStats.t_means[2])/kitti.transformStats.t_norm_mean
            self.trans_rot[i][3] = (edge[8] - kitti.transformStats.t_means[3])/kitti.transformStats.t_norm_mean
            self.trans_rot[i][4] = (edge[15] - kitti.transformStats.eul_means[1])/kitti.transformStats.eul_norm_mean
            self.trans_rot[i][5] = (edge[16] - kitti.transformStats.eul_means[2])/kitti.transformStats.eul_norm_mean
            self.trans_rot[i][6] = (edge[17] - kitti.transformStats.eul_means[3])/kitti.transformStats.eul_norm_mean
        elseif self.traintype == "trans_only" then
            self.translation[i][1] = (edge[6] - kitti.transformStats.t_means[1])/kitti.transformStats.t_norm_mean
            self.translation[i][2] = (edge[7] - kitti.transformStats.t_means[2])/kitti.transformStats.t_norm_mean
            self.translation[i][3] = (edge[8] - kitti.transformStats.t_means[3])/kitti.transformStats.t_norm_mean
        elseif self.traintype == "heatmap" then
            -- Generate heat map for spherical coordinates and Euler angles
            local az = edge[12]
            local el = edge[13]
            local norms = edge[14]
            -- Generate the heat map
            self.azHeat[i] = azHeatMapMaker(az)
            self.elHeat[i] = heatMapMaker(kitti.elHist,el)
            self.rhoHeat[i] = heatMapMaker(kitti.rhoHist,norms)
        end
end


----------------------------------------
-- Batch Factory
----------------------------------------
function BatchGenerator:makeNewBatch()
    local NewBatch = util.deepcopy(Batch)
    NewBatch.traintype = self.traintype
    NewBatch.batchSize = self.batchSize
    return NewBatch
end

function BatchGenerator:generateBatches(seqnos,edgeSet)
    print('Generating batches...')
    -- Initialize variables
    local Batch = self:makeNewBatch()
    -- Select random edges
    local allEdges = {}
    for p1,seqno in pairs(seqnos) do
        for p2,v in pairs(edgeSet[seqno][1]) do
            allEdges[#allEdges+1] = v
        end
    end

    local numEdges = math.min(#allEdges,
                        math.floor(self.nBatchesPerSet*self.batchSize))
    Batch.numEdges = numEdges
    local chosenEdges = torch.randperm(#allEdges):narrow(1,1,numEdges)
    -- Create Batch data
    Batch:initTestData(numEdges)
    Batch.data = torch.zeros(numEdges,2,kitti.imsize[1],kitti.imsize[2])

    for i=1,numEdges do
        -- Graph edges
        local seqno = allEdges[chosenEdges[i]][1]
        local k = allEdges[chosenEdges[i]][2]
        local edge = self.graphs[seqno][k]
        -- Grab image pair
        local imindex1 = edge[1]
        local imcamera1 = edge[2]+1
        local imindex2 = edge[3]
        local imcamera2 = edge[4]+1
        if imindex1 > self.imdata[seqno]:size(1) or imindex2 > self.imdata[seqno]:size(1) then
            error("IMAGE INDEX ERROR")
        end
        local im1 = self.imdata[seqno][{imindex1,imcamera1,{},{}}]
        local im2 = self.imdata[seqno][{imindex2,imcamera2,{},{}}]
        Batch.data[{i,1,{},{}}] = im1 + torch.randn(im1:size())*Batch.imsigma
        Batch.data[{i,2,{},{}}] = im2 + torch.randn(im2:size())*Batch.imsigma

        -- Get testing data (translation, rotation, heatmaps, etc.)
        Batch:getTestData(edge,i)

        -- Get rid of this edge in BatchGenerators edge
        edgeSet[seqno][1][k] = nil
        edgeSet[seqno][2] = edgeSet[seqno][2] - 1
    end

    return Batch
end

function BatchGenerator:generateTrainingBatches()
    return self:generateBatches(self.trainSeqNums,self.trainEdgeSet)
end

function BatchGenerator:generateInterTestingBatches()
    -- Generate new set every time
    return self:generateBatches(self.interTestSeqNums,self.createEdgeSet(self.interTestSeqNums))
end

function BatchGenerator:generateTestingBatches()
    return self:generateBatches(self.testSeqNums,self.testEdgeSet)
end

function Batch:hasNextBatch()
    return (self.index < self.numEdges)
end

function Batch:getNextBatch()
    if not self:hasNextBatch() then
        error("Error: Calling get next batch without checking if data is left")
    end
    local batch = {}
    local first = self.index
    local last = math.min(self.index + self.batchSize - 1, self.numEdges)
    if self.traintype == "regression" then
        batch.trans_rot = self.trans_rot[{{first,last},{}}]
    elseif self.traintype == "trans_only" then
        batch.translation = self.translation[{{first,last},{}}]
    elseif self.traintype == "heatmap" then
        batch.azHeat = self.azHeat[{{first,last},{}}]
        batch.elHeat = self.elHeat[{{first,last},{}}]
        batch.rhoHeat = self.rhoHeat[{{first,last},{}}]
    end
    batch.data = self.data[{{first,last},{},{},{}}]
    batch.numEdges = batch.data:size(1)
    self.index = self.index + self.batchSize
    return batch
end
