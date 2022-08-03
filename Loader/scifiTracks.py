# scifi geometry constants:
import numpy as np

# position of xuvx layers for each T1,T2,T3 tracking station
layerZ = np.array([7826.106,  7895.9189, 7966.1035, 8035.9048,
          8508.1064, 8577.8691, 8648.1543, 8717.9043,
          9193.1064, 9262.9824, 9333.041,  9402.9043])

# Tilt of each layer
layerDxDy = np.array([0, 0.0875, -0.0875, 0,
             0, 0.0875, -0.0875, 0,
             0, 0.0875, -0.0875, 0,
             0, 0.0875, -0.0875, 0])



class Hit:
    def __init__(self, data):
        self.x0 = data["x0"]
        self.z0 = data["z0"]
        
        for layer in range(12):

            #ensure hit is in a defined scifi layer
            assert int(self.z0) in layerZ.astype(int), "Unknown layer at z: "+str(self.z0)

            if int(layerZ[layer]) == int(self.z0):
                self.layer = layer
                self.dxdy = layerDxDy[layer]
                break



# A track in the scifi detector:
# Straight line in yz, but polynomial in xz to account for magnetic field
class ScifiTrack:
    def __init__(self, data):
        self.ax = data["ax"]
        self.bx = data["bx"]
        self.cx = data["cx"]
        self.ay = data["ay"]
        self.by = data["by"]
        self.hits = list(map(lambda x: Hit(x), data["hits"]))
        self.isGhost = data["ghost"]
        self.layers = [None] * 12
        for hit in self.hits:
            self.layers[hit.layer] = hit
        self.n_Hits= self.numHits()
        self.n_MissingHits= self.numMissingHits()
        self.chi_2 = self.chi2()

    def numHits(self):
        return len(self.hits)

    def numMissingHits(self):
        n = 0
        for layer in range(12):
            if self.layers[layer] != None: continue
            x = self.xAtZ(layerZ[layer])
            y = self.yAtZ(layerZ[layer])
            if abs(x) > 3000 or abs(y) > 2417.5: continue
            if (x**2 + y**2) < 81: continue
            n += 1

        return n

    def chi2(self):
        chi2 = 0
        for hit in self.hits:
            x = self.xAtZ(hit.z0)
            y = self.yAtZ(hit.z0)
            dx = x - (hit.x0 + y * hit.dxdy)
            chi2 += dx*dx
        return chi2

    def xAtZ(self, z):
        #polynomial function that fits the track in the bending (X-Z) plane
        #dz = z-z0
        # x(z) = ax + bx*(z-z0)+ cx *(z-z0)^2 * (1+ dRatio (z-z0))
        # where bx is the track slope dx/dz
        dz = z - 8520
        return self.ax + dz * (self.bx + dz * self.cx * (1 + -0.00028 * dz))

    def yAtZ(self, z):
        #linear function to model the track in the non-bending (Y-Z) plane
        # y(z) = ay + by * (z-z0)
        # where by is the track slope dy/dz
        return self.ay + self.by * (z - 8520)

    def txAtZ(self, z):
        dz = z - 8520
        return self.bx + 2 * dz * self.cx + 3 * dz * dz * self.cx * -0.00028
    
    def tyAtZ(self, z):
        return self.by
