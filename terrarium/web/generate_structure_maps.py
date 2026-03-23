import json
import math
import random
import os

def generate_structure_map(name, num_atoms, base_color, radius_scale):
    atoms = []
    bonds = []
    
    # Generate random points in a sphere or custom shape
    for i in range(num_atoms):
        u = random.random()
        v = random.random()
        theta = 2 * math.pi * u
        phi = math.acos(2 * v - 1)
        r = radius_scale * math.cbrt(random.random())
        
        x = r * math.sin(phi) * math.cos(theta)
        y = r * math.sin(phi) * math.sin(theta)
        z = r * math.cos(phi)
        
        # Color variation
        color = [
            min(255, max(0, base_color[0] + random.randint(-20, 20))),
            min(255, max(0, base_color[1] + random.randint(-20, 20))),
            min(255, max(0, base_color[2] + random.randint(-20, 20)))
        ]
        
        atoms.append({
            "id": i,
            "position": [x, y, z],
            "color": color,
            "element": "C" if random.random() > 0.3 else ("O" if random.random() > 0.5 else "H"),
            "radius": 0.5 + random.random() * 0.5
        })
        
    # Generate some bonds
    for i in range(num_atoms):
        # connect to 1-3 nearest neighbors
        num_bonds = random.randint(1, 3)
        distances = []
        for j in range(num_atoms):
            if i == j: continue
            dx = atoms[i]["position"][0] - atoms[j]["position"][0]
            dy = atoms[i]["position"][1] - atoms[j]["position"][1]
            dz = atoms[i]["position"][2] - atoms[j]["position"][2]
            dist = dx*dx + dy*dy + dz*dz
            distances.append((dist, j))
        
        distances.sort()
        for b in range(min(num_bonds, len(distances))):
            target = distances[b][1]
            if i < target: # avoid duplicate bonds
                bonds.append({"source": i, "target": target})
                
    return {"name": name, "atoms": atoms, "bonds": bonds}

def main():
    os.makedirs("terrarium/web/data/structure_maps", exist_ok=True)
    
    maps = {
        "plant": generate_structure_map("Plant Tissue", 500, [50, 200, 50], 10.0),
        "fish": generate_structure_map("Fish Scales", 600, [100, 150, 255], 12.0),
        "soil": generate_structure_map("Soil Minerals", 800, [139, 69, 19], 15.0),
        "water": generate_structure_map("Water Molecules", 400, [50, 100, 255], 20.0)
    }
    
    for key, data in maps.items():
        with open(f"terrarium/web/data/structure_maps/{key}.json", "w") as f:
            json.dump(data, f)
            print(f"Generated {key}.json")

if __name__ == "__main__":
    main()
