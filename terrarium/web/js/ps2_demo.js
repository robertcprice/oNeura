const canvas = document.getElementById('render-canvas');
const zoomStatus = document.getElementById('zoom-status');
const targetStatus = document.getElementById('target-status');
const loadingStatus = document.getElementById('loading-status');

// PS2 Internal Resolution
const PS2_WIDTH = 320;
const PS2_HEIGHT = 240;

// Setup Renderer
const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: "high-performance" });
renderer.setSize(PS2_WIDTH, PS2_HEIGHT, false); // False means don't set canvas CSS size
renderer.setPixelRatio(1);
renderer.outputEncoding = THREE.sRGBEncoding;

// Setup Scene & Camera
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a2b3c);
scene.fog = new THREE.FogExp2(0x1a2b3c, 0.05);

const camera = new THREE.PerspectiveCamera(60, PS2_WIDTH / PS2_HEIGHT, 0.1, 1000);
camera.position.set(0, 5, 10);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.maxDistance = 20;
controls.minDistance = 0.05;

// Lights
const ambient = new THREE.AmbientLight(0xffffff, 0.5);
scene.add(ambient);
const dirLight = new THREE.DirectionalLight(0xffddaa, 0.8);
dirLight.position.set(5, 10, 5);
scene.add(dirLight);

// Raycaster for targeting
const raycaster = new THREE.Raycaster();
const center = new THREE.Vector2(0, 0);

// Global State
let currentMode = 'MACRO'; // 'MACRO' or 'MOLECULAR'
let currentTarget = null;
let molecularGroup = new THREE.Group();
scene.add(molecularGroup);
molecularGroup.visible = false;

let macroGroup = new THREE.Group();
scene.add(macroGroup);

const structureCache = {};

// Helper: Add Macro Object
function createMacroObject(geometry, color, position, name, structureFile) {
    // Make it look a bit low-poly and flat shaded for PS2 vibe
    const material = new THREE.MeshPhongMaterial({ color: color, flatShading: true });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.copy(position);
    mesh.userData = { name, structureFile, originalColor: color };
    macroGroup.add(mesh);
    return mesh;
}

// Build Macro Scene (Terrarium / Aquarium)
function buildMacroScene() {
    // Soil
    createMacroObject(new THREE.BoxGeometry(10, 1, 10), 0x553311, new THREE.Vector3(0, -0.5, 0), "Soil bed", "soil");
    // Water
    const water = createMacroObject(new THREE.BoxGeometry(4, 0.5, 4), 0x2266cc, new THREE.Vector3(2, 0.25, 2), "Pond Water", "water");
    water.material.transparent = true;
    water.material.opacity = 0.8;
    // Plant
    createMacroObject(new THREE.CylinderGeometry(0.2, 0.2, 3, 5), 0x33aa33, new THREE.Vector3(-2, 1.5, -2), "Fern Plant", "plant");
    // Fish
    createMacroObject(new THREE.ConeGeometry(0.3, 1, 4), 0xff6600, new THREE.Vector3(2, 0.5, 2), "Koi Fish", "fish").rotation.z = Math.PI/2;
}

buildMacroScene();
loadingStatus.innerText = "Systems online. Ready.";
setTimeout(() => loadingStatus.innerText = "", 3000);

async function loadStructureMap(type) {
    if (structureCache[type]) return structureCache[type];
    
    loadingStatus.innerText = `Fetching ${type} map...`;
    try {
        const res = await fetch(`data/structure_maps/${type}.json`);
        const data = await res.json();
        structureCache[type] = data;
        loadingStatus.innerText = "";
        return data;
    } catch(e) {
        console.error("Failed to load map:", e);
        loadingStatus.innerText = "Error loading map";
        return null;
    }
}

function buildMolecularView(data, centerPosition) {
    // Clear old
    while(molecularGroup.children.length > 0) { 
        molecularGroup.remove(molecularGroup.children[0]); 
    }
    
    if (!data) return;
    
    // Scale down the molecular view so it fits inside the zoomed-in macro object
    const scale = 0.05; 
    
    // Atoms
    const atomGeo = new THREE.IcosahedronGeometry(1, 1); // low poly sphere
    const materialMap = {};
    
    data.atoms.forEach(atom => {
        const colorHex = (atom.color[0] << 16) | (atom.color[1] << 8) | atom.color[2];
        if (!materialMap[colorHex]) {
            materialMap[colorHex] = new THREE.MeshPhongMaterial({ color: colorHex, flatShading: true });
        }
        
        const mesh = new THREE.Mesh(atomGeo, materialMap[colorHex]);
        mesh.position.set(
            centerPosition.x + atom.position[0] * scale,
            centerPosition.y + atom.position[1] * scale,
            centerPosition.z + atom.position[2] * scale
        );
        mesh.scale.setScalar(atom.radius * scale * 0.5);
        molecularGroup.add(mesh);
    });
    
    // Bonds
    const lineMat = new THREE.LineBasicMaterial({ color: 0x888888 });
    const points = [];
    data.bonds.forEach(bond => {
        const a = data.atoms[bond.source];
        const b = data.atoms[bond.target];
        if(!a || !b) return;
        points.push(new THREE.Vector3(
            centerPosition.x + a.position[0] * scale,
            centerPosition.y + a.position[1] * scale,
            centerPosition.z + a.position[2] * scale
        ));
        points.push(new THREE.Vector3(
            centerPosition.x + b.position[0] * scale,
            centerPosition.y + b.position[1] * scale,
            centerPosition.z + b.position[2] * scale
        ));
    });
    const lineGeo = new THREE.BufferGeometry().setFromPoints(points);
    const lines = new THREE.LineSegments(lineGeo, lineMat);
    molecularGroup.add(lines);
}

// Animation Loop
let lastTime = 0;
function animate(time) {
    requestAnimationFrame(animate);
    
    const dt = (time - lastTime) / 1000;
    lastTime = time;
    
    controls.update();

    // Fish animation
    macroGroup.children.forEach(c => {
        if (c.userData.name === "Koi Fish") {
            c.position.x = 2 + Math.sin(time/500) * 1.5;
            c.position.z = 2 + Math.cos(time/500) * 1.5;
            c.rotation.y = -(time/500) + Math.PI;
        }
    });

    // Check target in center of screen
    raycaster.setFromCamera(center, camera);
    const intersects = raycaster.intersectObjects(macroGroup.children);
    
    let hit = null;
    if (intersects.length > 0) {
        hit = intersects[0].object;
    }
    
    // Highlight logic
    macroGroup.children.forEach(c => {
        if (c === hit && currentMode === 'MACRO') {
            c.material.emissive.setHex(0x333333);
        } else {
            c.material.emissive.setHex(0x000000);
        }
    });

    if (hit) {
        currentTarget = hit.userData;
        targetStatus.innerText = "Target: " + currentTarget.name;
    } else {
        currentTarget = null;
        targetStatus.innerText = "Target: None";
    }
    
    // Distance check for zoom
    let dist = camera.position.distanceTo(controls.target);
    
    if (dist < 1.5 && hit && currentMode === 'MACRO') {
        // Transition to Molecular
        currentMode = 'MOLECULAR';
        zoomStatus.innerText = "Scale: MOLECULAR STRUCTURE";
        zoomStatus.style.color = "#00ffff";
        macroGroup.visible = false;
        molecularGroup.visible = true;
        scene.fog.density = 0.5; // denser fog inside
        
        loadStructureMap(hit.userData.structureFile).then(data => {
            if (data && currentMode === 'MOLECULAR') {
                buildMolecularView(data, hit.position);
            }
        });
    } else if (dist >= 1.5 && currentMode === 'MOLECULAR') {
        // Transition back to Macro
        currentMode = 'MACRO';
        zoomStatus.innerText = "Scale: MACRO ECOSYSTEM";
        zoomStatus.style.color = "#0f0";
        macroGroup.visible = true;
        molecularGroup.visible = false;
        scene.fog.density = 0.05;
    }

    if (currentMode === 'MOLECULAR') {
        // slow rotation of atoms
        molecularGroup.rotation.y += 0.5 * dt;
    } else {
        molecularGroup.rotation.y = 0;
    }

    // Optional vertex snapping in shader would be cool, but pixelated render target gives strong PS2 vibe
    // Ensure viewport scales properly
    
    renderer.render(scene, camera);
}

animate(0);
