require 'torch'
require 'image'
require 'paths'
local matio = require 'matio'

------------------------------------------------------
-- Pose graph functions
------------------------------------------------------
posegraph = {}
posegraph.location = "./PoseGraph/"
posegraph.posesSeqs = matio.load(posegraph.location .. "poses.mat","P")
posegraph.camOffset = { torch.Tensor({0, 0, 0}),
                           torch.Tensor({((3.798145e+02)/(7.070912e+02)), 0, 0})};

function getExpCoordinates(dR)
    local omega = torch.Tensor({dR[3][2] - dR[2][3],
                                dR[1][3] - dR[3][1],
                                dR[2][1] - dR[1][2]}):div(2)
    return omega:mul(math.asin(torch.norm(omega))/torch.norm(omega))
end

function posegraph.graphLocation(seqno)
    return paths.concat(posegraph.location, string.format('seq%02dGraph.mat',seqno))
end


-- This function slightly different from the matlab one
function posegraph.createPoseGraphs(seqno,force)
    -- Note that the naming conventions here for polar coordinates are a bit misleading.
    -- What's called "el" (for elevation) is actually the inclination about the yz plane.
    -- What's called "az" (for azimuth) is the rotation about the x-axis (what would normally
    -- be called the elevation).
    -- rho is the norm of the heading as usual.
    force = force or false -- Force it to make a new one, no by default
    assert(0 <= seqno and seqno <= #kitti.seqSizes-1, 'Seqno is not in range 0 to 10')
    local graphloc = posegraph.graphLocation(seqno) -- get dataset location
    if paths.filep(graphloc) and not force then
        return -- Already made - no need to remake it
    end

    local numi = 0
    -- constants
    local k1 = 10
    local k2 = 1
    local thresh = 10
    -- Begin graph creation
    local numposes = posegraph.posesSeqs[seqno+1]:size(1)
    local GraphTable = torch.Tensor(220000,17)
    local total = 1

    for i=1,numposes do
        local j = 1
        local Pose1 = torch.reshape(posegraph.posesSeqs[seqno+1][i],3,4)
                        :type(torch.getdefaulttensortype())
        local R1 = Pose1[{{1,3},{1,3}}]
        local T1 = Pose1[{{1,3},{4}}]
        -- Compute spherical coordinates (radius, inclination, and azimuth)
        while i + j <= numposes do
            local Pose2 = torch.reshape(posegraph.posesSeqs[seqno+1][i+j],3,4)
                            :type(torch.getdefaulttensortype())
            local R2 = Pose2[{{1,3},{1,3}}]
            local T2 = Pose2[{{1,3},{4}}]
            local dR = R1:t()*R2
            -- Compute Euler Angles
            local eul1 =  math.atan2(dR[3][2], dR[3][3]) -- psi
            local eul2 = -math.asin(dR[3][1]) -- theta
            local eul3 =  math.atan2(dR[2][1], dR[1][1]) -- phi
            -- Get exponential coordinates
            local omega = getExpCoordinates(dR)
            -- Check if distance is too great
            if k1*torch.norm(omega) + k2*torch.norm(R1:t()*(T2-T1)) > thresh then
                break
            end
            for cam1=1,2 do
                for cam2=1,2 do
                    dT = R1:t()*((T2+posegraph.camOffset[cam1])-(T1+posegraph.camOffset[cam2]))
                    dT = dT[{{},1}]
                    -- Compute spherical coordinates (in twisted frame)
                    local rho = torch.norm(dT)
                    local el = math.atan2(-dT[1],math.sqrt(dT[2]^2+dT[3]^2))
                    local az = math.atan2(dT[2],dT[3])
                    -- Store everything
                    -- TODO: Cameras zero index for compatibility reasons, make it 1 indexed
                    GraphTable[total]:copy(torch.Tensor(
                        {i,cam1-1,i+j,cam2-1,seqno,
                         dT[1],dT[2],dT[3],
                         omega[1],omega[2],omega[3],
                         az,el,rho,
                         eul1,eul2,eul3}))
                    total = total + 1
                end
            end
            j = j + 1
        end
        numi = numi + 1
    end
    total = total - 1 -- overcounted by 1
    print("Seq " .. seqno .. " Total done: " .. total)
    print("Average degree: " .. total/numi)
    matio.save(graphloc, GraphTable[{{1,total},{}}])
end
