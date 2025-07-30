// import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.152.2/build/three.module.js';
// import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.152.2/examples/jsm/controls/OrbitControls.js';

import * as THREE from 'three'
// import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
let scene = new THREE.Scene();
let camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
let renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);
let controls = new OrbitControls(camera, renderer.domElement);
camera.position.set(0, 0, 50);
controls.update();

let allObjects = [];
let linesGroup = new THREE.Group();
let inliersGroup = new THREE.Group();
scene.add(linesGroup);
scene.add(inliersGroup);

function clearGroup(group) {
    while (group.children.length) group.remove(group.children[0]);
}

function addPoints(points, color = 0x888888, size = 0.2) {
    let geometry = new THREE.BufferGeometry();
    let positions = [];
    for (let p of points) positions.push(p.x, p.y, p.z);
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    let material = new THREE.PointsMaterial({ color, size });
    let pts = new THREE.Points(geometry, material);
    scene.add(pts);
    allObjects.push(pts);
}

function addLine(p0, p1, color = 0x0000ff, width = 2) {
    let geometry = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(p0.x, p0.y, p0.z),
        new THREE.Vector3(p1.x, p1.y, p1.z)
    ]);
    let material = new THREE.LineBasicMaterial({ color, linewidth: width });
    let line = new THREE.Line(geometry, material);
    linesGroup.add(line);
}

function addInliers(points, color = 0x00ff00, size = 0.4) {
    let geometry = new THREE.BufferGeometry();
    let positions = [];
    for (let p of points) positions.push(p.x, p.y, p.z);
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    let material = new THREE.PointsMaterial({ color, size });
    let pts = new THREE.Points(geometry, material);
    inliersGroup.add(pts);
}

function updateDisplay(data, threshold) {
    clearGroup(linesGroup);
    clearGroup(inliersGroup);
    for (let line of data.lines) {
        if (line.inliers >= threshold) {
            let factor = 30;
            //plot linep0 linep1 extend to  factor the length of the line
            let lineLength = new THREE.Vector3(line.lineP1.x - line.lineP0.x, line.lineP1.y - line.lineP0.y, line.lineP1.z - line.lineP0.z).length();
            let direction = new THREE.Vector3(line.lineP1.x - line.lineP0.x, line.lineP1.y - line.lineP0.y, line.lineP1.z - line.lineP0.z).normalize();
            let extendedP1 = new THREE.Vector3(
                line.lineP1.x + direction.x * factor * lineLength,
                line.lineP1.y + direction.y * factor * lineLength,
                line.lineP1.z + direction.z * factor * lineLength
            );
            addLine(line.lineP0, extendedP1, 0xff0000, 2);
            addLine(line.p0, line.p1, 0xffff00, 2);
            addInliers(line.inlierCloud, 0x00ff00, 0.4);
        }
    }
}



let currentData = null; // Store current loaded data

// Function to load and display a specific JSON file
function loadJSONFile(filename) {
    fetch(`/json/${filename}`)
        .then(r => r.json())
        .then(data => {
            // Clear existing objects
            for (let obj of allObjects) {
                scene.remove(obj);
            }
            allObjects = [];
            clearGroup(linesGroup);
            clearGroup(inliersGroup);

            currentData = data;
            addPoints(data.points, 0x888888, 0.2);
            updateDisplay(data, parseInt(document.getElementById('inlierSlider').value));
            document.getElementById('inlierSlider').max = Math.max(...data.lines.map(l => l.inliers));

            // Update current file indicator
            document.getElementById('currentFile').textContent = filename;
        })
        .catch(err => {
            console.error('Failed to load file:', filename, err);
            alert(`Failed to load ${filename}`);
        });
}

// Function to load available JSON files
function loadFileList() {
    fetch('json/')
        .then(r => r.text())
        .then(html => {
            // Parse directory listing HTML to extract .json files
            const parser = new DOMParser();
            const doc = parser.parseFromString(html, 'text/html');
            const links = doc.querySelectorAll('a');
            const jsonFiles = [];

            links.forEach(link => {
                const href = link.getAttribute('href');
                if (href && href.endsWith('.json')) {
                    jsonFiles.push(href);
                }
            });

            // Populate file list
            const filesDiv = document.getElementById('files');
            filesDiv.innerHTML = '';

            if (jsonFiles.length === 0) {
                filesDiv.innerHTML = '<em>No JSON files found</em>';
                return;
            }

            jsonFiles.forEach(filename => {
                const fileButton = document.createElement('div');
                fileButton.className = 'file-item';
                fileButton.textContent = filename;
                fileButton.onclick = () => loadJSONFile(filename);
                filesDiv.appendChild(fileButton);
            });

            // Auto-load first file if available
            if (jsonFiles.length > 0) {
                loadJSONFile(jsonFiles[0]);
            }
        })
        .catch(err => {
            console.error('Failed to load file list:', err);
            document.getElementById('files').innerHTML = '<em>Failed to load file list</em>';
        });
}

// Update slider event listener to use currentData
document.getElementById('inlierSlider').addEventListener('input', function () {
    document.getElementById('inlierValue').textContent = this.value;
    if (currentData) {
        updateDisplay(currentData, parseInt(this.value));
    }
});

// Initialize file list on page load
loadFileList();


// fetch('results.json')
//     .then(r => r.json())
//     .then(data => {
//         addPoints(data.points, 0x888888, 0.2);
//         updateDisplay(data, parseInt(document.getElementById('inlierSlider').value));
//         document.getElementById('inlierSlider').max = Math.max(...data.lines.map(l => l.inliers));
//         document.getElementById('inlierSlider').addEventListener('input', function () {
//             document.getElementById('inlierValue').textContent = this.value;
//             updateDisplay(data, parseInt(this.value));
//         });
//     });

window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}
animate();