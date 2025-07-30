// Next Frame button logic
document.getElementById('nextFrameButton').addEventListener('click', function () {
    if (!sceneData || sceneData.length === 0) return;
    const sequenceSelect = document.getElementById('sequenceSelect');
    const frameSelect = document.getElementById('frameSelect');
    const seqIds = Object.keys(sceneData.reduce((acc, f) => { acc[f.seq_id] = true; return acc; }, {})).sort((a, b) => parseInt(a) - parseInt(b));
    let currentSeq = sequenceSelect.value;
    let currentFrame = frameSelect.value;
    // Find current sequence and frame index
    let seqIdx = seqIds.indexOf(currentSeq);
    let framesInSeq = sceneData.filter(f => f.seq_id == currentSeq).sort((a, b) => a.frame_id - b.frame_id);
    let frameIdx = framesInSeq.findIndex(f => f.frame_id == currentFrame);

    // Advance to next frame
    if (frameIdx < framesInSeq.length - 1) {
        // Next frame in current sequence
        frameSelect.value = framesInSeq[frameIdx + 1].frame_id;
        loadSceneFrame(framesInSeq[frameIdx + 1]);
    } else {
        // Last frame in this sequence
        if (seqIdx < seqIds.length - 1) {
            // Go to first frame of next sequence
            let nextSeq = seqIds[seqIdx + 1];
            sequenceSelect.value = nextSeq;
            const nextFrames = sceneData.filter(f => f.seq_id == nextSeq).sort((a, b) => a.frame_id - b.frame_id);
            frameSelect.innerHTML = '';
            nextFrames.forEach(frame => {
                const option = document.createElement('option');
                option.value = frame.frame_id;
                option.textContent = `Frame ${frame.frame_id}`;
                option.frameData = frame;
                frameSelect.appendChild(option);
            });
            frameSelect.value = nextFrames[0].frame_id;
            loadSceneFrame(nextFrames[0]);
        } else {
            // Last frame in last sequence: go to first frame of first sequence
            let firstSeq = seqIds[0];
            sequenceSelect.value = firstSeq;
            const firstFrames = sceneData.filter(f => f.seq_id == firstSeq).sort((a, b) => a.frame_id - b.frame_id);
            frameSelect.innerHTML = '';
            firstFrames.forEach(frame => {
                const option = document.createElement('option');
                option.value = frame.frame_id;
                option.textContent = `Frame ${frame.frame_id}`;
                option.frameData = frame;
                frameSelect.appendChild(option);
            });
            frameSelect.value = firstFrames[0].frame_id;
            loadSceneFrame(firstFrames[0]);
        }
    }
});



import * as THREE from 'three'
// import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
let scene = new THREE.Scene();

// Add axis marker at the origin with graduated marks
const axesLength = 100;
const axesHelper = new THREE.AxesHelper(axesLength);
scene.add(axesHelper);

// Add graduated marks (ticks) every 10 units on each axis
const tickLength = 4;
const tickColor = 0x888888;
const tickMaterial = new THREE.LineBasicMaterial({ color: tickColor });
const tickGroup = new THREE.Group();
for (let i = 10; i < axesLength; i += 10) {
    // X axis ticks (red)
    let xTickGeom = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(i, -tickLength, 0),
        new THREE.Vector3(i, tickLength, 0)
    ]);
    let xTick = new THREE.Line(xTickGeom, tickMaterial);
    tickGroup.add(xTick);
    // Negative X
    let xTickNegGeom = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(-i, -tickLength, 0),
        new THREE.Vector3(-i, tickLength, 0)
    ]);
    let xTickNeg = new THREE.Line(xTickNegGeom, tickMaterial);
    tickGroup.add(xTickNeg);

    // Y axis ticks (green)
    let yTickGeom = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(-tickLength, i, 0),
        new THREE.Vector3(tickLength, i, 0)
    ]);
    let yTick = new THREE.Line(yTickGeom, tickMaterial);
    tickGroup.add(yTick);
    // Negative Y
    let yTickNegGeom = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(-tickLength, -i, 0),
        new THREE.Vector3(tickLength, -i, 0)
    ]);
    let yTickNeg = new THREE.Line(yTickNegGeom, tickMaterial);
    tickGroup.add(yTickNeg);

    // Z axis ticks (blue)
    let zTickGeom = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0, -tickLength, i),
        new THREE.Vector3(0, tickLength, i)
    ]);
    let zTick = new THREE.Line(zTickGeom, tickMaterial);
    tickGroup.add(zTick);
    // Negative Z
    let zTickNegGeom = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0, -tickLength, -i),
        new THREE.Vector3(0, tickLength, -i)
    ]);
    let zTickNeg = new THREE.Line(zTickNegGeom, tickMaterial);
    tickGroup.add(zTickNeg);
}
scene.add(tickGroup);
let camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
let cameraDistance = 200;
// Default: point at 0 deg (positive Z)
camera.position.set(0, -cameraDistance, -cameraDistance);
camera.lookAt(0, 0, 0);
scene.rotation.set(0, 0, Math.PI / 2); // Rotate 45 degrees around Y axis

let renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);
let controls = new OrbitControls(camera, renderer.domElement);
controls.update();

let linesGroup = new THREE.Group();
let inliersGroup = new THREE.Group();
let validPairsGroup = new THREE.Group();
scene.add(linesGroup);
scene.add(inliersGroup);
validPairsGroup.name = 'ValidLinePairs';
scene.add(validPairsGroup);



let sceneData = null; // Store the MAN scene data
let radarGroups = {}; // Store radar sensor groups
let currentFrameData = null; // Store current frame data
let currentRadarData = {}; // Store raw radar data for coordinate transformations
let currentValidPairs = []; // Store current valid pairs

// Initialize radar groups
const radarSensors = ['RADAR_LEFT_FRONT', 'RADAR_LEFT_BACK', 'RADAR_RIGHT_FRONT',
    'RADAR_RIGHT_BACK', 'RADAR_LEFT_SIDE', 'RADAR_RIGHT_SIDE'];
radarSensors.forEach(sensor => {
    radarGroups[sensor] = new THREE.Group();
    radarGroups[sensor].name = sensor;
    scene.add(radarGroups[sensor]);
});

// Radar colors for differentiation
const radarColors = {
    'RADAR_LEFT_FRONT': 0xff0000,   // Red
    'RADAR_LEFT_BACK': 0x00ff00,    // Green
    'RADAR_RIGHT_FRONT': 0x0000ff,  // Blue
    'RADAR_RIGHT_BACK': 0xffff00,   // Yellow
    'RADAR_LEFT_SIDE': 0xff00ff,    // Magenta
    'RADAR_RIGHT_SIDE': 0x00ffff    // Cyan
};

// --- Camera display logic ---
const cameraChannels = [
    'CAMERA_LEFT_FRONT',
    'CAMERA_LEFT_BACK',
    'CAMERA_RIGHT_FRONT',
    'CAMERA_RIGHT_BACK'
];
let cameraPlanes = [];

function clearCameraPlanes() {
    for (const plane of cameraPlanes) {
        scene.remove(plane);
    }
    cameraPlanes = [];
}

function addAllCameraPlanes(frameData, distance = 200) {
    clearCameraPlanes();
    for (const channel of cameraChannels) {
        const cam = frameData[channel];
        if (cam && cam.image_file && cam.rotation && cam.translation && cam.intrinsics) {
            addCameraImagePlane(cam, channel, distance);
        }
    }
}
function addCameraImagePlane(cameraInfo, name = "CAMERA", dist_from_cam = 200) {
    if (!cameraInfo || !cameraInfo.image_file || !cameraInfo.intrinsics || !cameraInfo.rotation || !cameraInfo.translation) {
        console.warn("Camera info missing required fields");
        return;
    }

    // Load the image
    const loader = new THREE.TextureLoader();
    loader.load(cameraInfo.image_file, function (texture) {
        // Get image aspect ratio from texture
        const aspect = texture.image.width / texture.image.height;
        // Use intrinsics to set the size of the plane (optional: scale by focal length)
        const x_focal_length = cameraInfo.intrinsics[0][0]; // Assuming intrinsics is a 3x3 matrix
        const width = texture.image.width * dist_from_cam / x_focal_length
        const height = width / aspect

        const geometry = new THREE.PlaneGeometry(width, height);
        const material = new THREE.MeshBasicMaterial({ map: texture, side: THREE.DoubleSide });
        const plane = new THREE.Mesh(geometry, material);
        plane.name = name + "_image_plane";

        const rot = cameraInfo.rotation;

        // 1. Compute world position for (0,0,dist_from_cam) in camera coords
        const camZ = new THREE.Vector3(0, 0, dist_from_cam);
        console.log(`Adding camera plane for ${name} at distance ${dist_from_cam}`);

        // Build R as Matrix3
        const R = new THREE.Matrix3();
        R.set(
            rot[0][0], rot[0][1], rot[0][2],
            rot[1][0], rot[1][1], rot[1][2],
            rot[2][0], rot[2][1], rot[2][2]
        );

        // Apply R to camZ
        const worldOffset = camZ.clone().applyMatrix3(R);

        // Add t
        const t = new THREE.Vector3(
            cameraInfo.translation[0],
            cameraInfo.translation[1],
            cameraInfo.translation[2]
        );
        const worldPos = worldOffset.add(t);
        plane.position.copy(worldPos);

        // 2. Set orientation to R
        const matrix = new THREE.Matrix4();
        matrix.makeBasis(
            new THREE.Vector3(rot[0][0], rot[1][0], rot[2][0]), // X
            new THREE.Vector3(rot[0][1], rot[1][1], rot[2][1]), // Y
            new THREE.Vector3(rot[0][2], rot[1][2], rot[2][2])  // Z
        );
        plane.setRotationFromMatrix(matrix);


        scene.add(plane);
    });
}

// Function to clear valid pairs visualization
function clearValidPairs() {
    while (validPairsGroup.children.length) {
        validPairsGroup.remove(validPairsGroup.children[0]);
    }
}

// Function to visualize valid line pairs
function visualizeValidPairs(validPairs) {
    clearValidPairs();


    if (!document.getElementById('showOnlyValidPairs').checked) {
        analysisDiv.innerHTML = '<em>Show ONLY Valid Pairs is not enabled.</em>';
        return;
    }

    // 3D visualization as before
    validPairs.forEach((pair, index) => {
        // Create connection line between centroids
        const geometry = new THREE.BufferGeometry().setFromPoints([
            pair.centroid1,
            pair.centroid2,
        ]);

        // Use a bright color for connection lines (white/yellow)
        const material = new THREE.LineBasicMaterial({
            color: 0xffff00,
            linewidth: 3,
            transparent: true,
            opacity: 0.8
        });

        const connectionLine = new THREE.Line(geometry, material);
        validPairsGroup.add(connectionLine);

        // Add small spheres at centroids to highlight them
        const sphereGeometry = new THREE.SphereGeometry(0.2, 8, 6);
        const sphereMaterial = new THREE.MeshBasicMaterial({ color: 0xffff00 });

        const sphere1 = new THREE.Mesh(sphereGeometry, sphereMaterial);
        sphere1.position.copy(pair.centroid1);
        validPairsGroup.add(sphere1);

        const sphere2 = new THREE.Mesh(sphereGeometry, sphereMaterial);
        sphere2.position.copy(pair.centroid2);
        validPairsGroup.add(sphere2);

        const sphereMaterial2 = new THREE.MeshBasicMaterial({ color: 0xffaa00 });
        const sphere3 = new THREE.Mesh(sphereGeometry, sphereMaterial2);
        sphere3.position.copy(pair.projection1);
        validPairsGroup.add(sphere3);
        const sphere4 = new THREE.Mesh(sphereGeometry, sphereMaterial2);
        sphere4.position.copy(pair.projection2);
        validPairsGroup.add(sphere4);
    });

    console.log(`Visualized ${validPairs.length} valid line pairs`);
}
// Function to check if a line is part of valid pairs
function isLineInValidPairs(radar, lineIndex) {
    if (!document.getElementById('showOnlyValidPairs').checked) {
        return true; // Show all lines if not filtering
    }

    return currentValidPairs.some(pair =>
        (pair.radar1 === radar && pair.line1Index === lineIndex) ||
        (pair.radar2 === radar && pair.line2Index === lineIndex)
    );
}
// Function to calculate line direction vector
function getLineDirection(line) {
    const dir = new THREE.Vector3(
        line.lineP1.x - line.lineP0.x,
        line.lineP1.y - line.lineP0.y,
        line.lineP1.z - line.lineP0.z
    );
    return dir.normalize();
}

// Function to calculate line centroid
function getLineCentroid(line) {
    return new THREE.Vector3(
        line.lineP0.x, line.lineP0.y, line.lineP0.z);
}

// Function to calculate angle between two vectors in degrees
function getAngleBetweenVectors(v1, v2) {
    const dot = v1.dot(v2);
    const angle = Math.acos(Math.max(-1, Math.min(1, dot))); // Clamp to avoid numerical errors
    return (angle * 180) / Math.PI;
}

// Function to calculate distance from point to line
function getDistanceFromPointToLineOld(point, lineStart, lineEnd) {
    const lineVec = new THREE.Vector3().subVectors(lineEnd, lineStart);
    const pointVec = new THREE.Vector3().subVectors(point, lineStart);

    const lineLengthSq = lineVec.lengthSq();
    if (lineLengthSq === 0) return point.distanceTo(lineStart);

    const t = Math.max(0, Math.min(1, pointVec.dot(lineVec) / lineLengthSq));
    const projection = new THREE.Vector3().addVectors(lineStart, lineVec.multiplyScalar(t));

    return [point.distanceTo(projection), projection]; // Return distance and projection point
}

//funtion to project a point onto a line, the line represented by two points
function getDistanceFromPointToLine(point, lineStart, lineEnd) {
    const lineVec = new THREE.Vector3().subVectors(lineEnd, lineStart);
    const pointVec = new THREE.Vector3().subVectors(point, lineStart);

    const lineLengthSq = lineVec.lengthSq();
    if (lineLengthSq === 0) //two points are the same, return null,null
        return [null, null];

    //projection = lineV . pointV / lineLengthSq
    const projectedLength = lineVec.dot(pointVec) / lineLengthSq;
    const projection = new THREE.Vector3().addVectors(lineStart, lineVec.multiplyScalar(projectedLength));

    return [point.distanceTo(projection), projection];
}

// Function to analyze line pairs across different radars
function analyzeLinePairs() {
    const angleThreshold = parseFloat(document.getElementById('angleSlider').value);
    const spatialThreshold = parseFloat(document.getElementById('spatialSlider').value);
    const inlierThreshold = parseInt(document.getElementById('inlierSlider').value);

    console.log(`\n=== Line Pair Analysis ===`);
    console.log(`Angle threshold: ${angleThreshold}°`);
    console.log(`Spatial threshold: ${spatialThreshold}m`);
    console.log(`Inlier threshold: ${inlierThreshold}`);

    const validLinePairs = [];
    const radarSensorPairs = [];

    // Generate all radar sensor pairs
    for (let i = 0; i < radarSensors.length; i++) {
        for (let j = i + 1; j < radarSensors.length; j++) {
            radarSensorPairs.push([radarSensors[i], radarSensors[j]]);
        }
    }

    let totalPairs = 0;
    let filteredPairs = 0;

    // Analyze each radar pair
    radarSensorPairs.forEach(([radar1, radar2]) => {
        if (!currentRadarData[radar1] || !currentRadarData[radar2]) return;

        // Get display data (transformed if needed)
        let data1 = currentRadarData[radar1];
        let data2 = currentRadarData[radar2];

        if (getCurrentCoordSystem() === 'world' && currentFrameData) {
            if (currentFrameData[radar1]) {
                data1 = transformDataToWorld(currentRadarData[radar1], currentFrameData[radar1]);
            }
            if (currentFrameData[radar2]) {
                data2 = transformDataToWorld(currentRadarData[radar2], currentFrameData[radar2]);
            }
        }

        // Filter lines by inlier threshold
        const lines1 = data1.lines.filter(line => line.inliers >= inlierThreshold);
        const lines2 = data2.lines.filter(line => line.inliers >= inlierThreshold);

        // Compare all line pairs between these radars
        lines1.forEach((line1, idx1) => {
            lines2.forEach((line2, idx2) => {
                totalPairs++;

                // Calculate line directions
                const dir1 = getLineDirection(line1);
                const dir2 = getLineDirection(line2);

                // Calculate angle between lines (consider both orientations)
                const angle1 = getAngleBetweenVectors(dir1, dir2);
                const angle2 = getAngleBetweenVectors(dir1, dir2.clone().negate());
                const minAngle = Math.min(angle1, angle2);

                // Calculate centroids
                const centroid1 = getLineCentroid(line1);
                const centroid2 = getLineCentroid(line2);

                // Calculate spatial distance (distance from centroid to the other line)
                const [dist1, projection1] = getDistanceFromPointToLine(
                    centroid1,
                    new THREE.Vector3(line2.lineP0.x, line2.lineP0.y, line2.lineP0.z),
                    new THREE.Vector3(line2.lineP1.x, line2.lineP1.y, line2.lineP1.z)
                );
                const [dist2, projection2] = getDistanceFromPointToLine(
                    centroid2,
                    new THREE.Vector3(line1.lineP0.x, line1.lineP0.y, line1.lineP0.z),
                    new THREE.Vector3(line1.lineP1.x, line1.lineP1.y, line1.lineP1.z)
                );
                const spatialDistance = Math.min(dist1, dist2); // or use (dist1 + dist2) / 2 for average
                // Check if pair meets thresholds
                if (minAngle <= angleThreshold && spatialDistance <= spatialThreshold) {
                    filteredPairs++;

                    const pairInfo = {
                        radar1: radar1,
                        radar2: radar2,
                        line1Index: idx1,
                        line2Index: idx2,
                        angle: minAngle,
                        spatialDistance: spatialDistance,
                        projection1: projection1,
                        projection2: projection2,
                        line1Inliers: line1.inliers,
                        line2Inliers: line2.inliers,
                        centroid1: centroid1,
                        centroid2: centroid2
                    };

                    validLinePairs.push(pairInfo);
                }
            });
        });
    });

    console.log(`\nTotal line pairs examined: ${totalPairs}`);
    console.log(`Filtered pairs meeting criteria: ${filteredPairs}`);
    console.log(`\nValid line pairs:`);

    validLinePairs.forEach((pair, index) => {
        console.log(`${index + 1}. ${pair.radar1} L${pair.line1Index} ↔ ${pair.radar2} L${pair.line2Index}`);
        console.log(`   Angle: ${pair.angle.toFixed(2)}°, Distance: ${pair.spatialDistance.toFixed(2)}m`);
        console.log(`   Inliers: ${pair.line1Inliers} & ${pair.line2Inliers}`);
        console.log(`   Centroids: (${pair.centroid1.x.toFixed(2)}, ${pair.centroid1.y.toFixed(2)}, ${pair.centroid1.z.toFixed(2)}) ↔ (${pair.centroid2.x.toFixed(2)}, ${pair.centroid2.y.toFixed(2)}, ${pair.centroid2.z.toFixed(2)})`);
        console.log(`   Projections: (${pair.projection1.x.toFixed(2)}, ${pair.projection1.y.toFixed(2)}, ${pair.projection1.z.toFixed(2)}) ↔ (${pair.projection2.x.toFixed(2)}, ${pair.projection2.y.toFixed(2)}, ${pair.projection2.z.toFixed(2)})`);
    });

    currentValidPairs = validLinePairs;
    visualizeValidPairs(validLinePairs);
    refreshRadarDisplay();
    updateAnalysisResultsTable(validLinePairs);
    return validLinePairs;
    // Fill the analysis_results div with a table of valid line pairs
    function updateAnalysisResultsTable(validPairs) {

        // Table for analysis results
        const analysisDiv = document.getElementById('analysis_results');
        if (!analysisDiv) return;
        const container = document.getElementById('analysis_results');
        if (!container) return;
        if (!validPairs || validPairs.length === 0) {
            container.innerHTML = `<b>Analysis Results:</b> <span style="color:#888">No valid pairs found.</span>`;
            return;
        }



        // Build HTML table with seq_id and frame_id in every row, and add CSV copy button
        let html = `<div style="margin-bottom:4px;font-size:14px;">
        <b>Sequence:</b> <span style="color:#005">${currentFrameData && currentFrameData.seq_id !== undefined ? currentFrameData.seq_id : '-'}</span>
        &nbsp; <b>Frame:</b> <span style="color:#005">${currentFrameData && currentFrameData.frame_id !== undefined ? currentFrameData.frame_id : '-'}</span>
    </div>`;
        html += `<button id="copyPairsCsvBtn" style="font-size:12px;padding:2px 8px;margin-bottom:4px;">Copy as CSV</button>&nbsp;`;
        html += `<button id="copyPairsJsonBtn" style="font-size:12px;padding:2px 8px;margin-bottom:4px;">Copy as JSON</button>&nbsp;`;
        html += `<table style="border-collapse:collapse;font-size:13px;">
        <thead><tr style="background:#eee;">
            <th style="border:1px solid #ccc;padding:2px 6px;">#</th>
            <th style="border:1px solid #ccc;padding:2px 6px;">Seq</th>
            <th style="border:1px solid #ccc;padding:2px 6px;">Frm</th>
            <th style="border:1px solid #ccc;padding:2px 6px;">R1</th>
            <th style="border:1px solid #ccc;padding:2px 6px;">Id1</th>
            <th style="border:1px solid #ccc;padding:2px 6px;">Count1</th>
            <th style="border:1px solid #ccc;padding:2px 6px;">R2</th>
            <th style="border:1px solid #ccc;padding:2px 6px;">Id2</th>
            <th style="border:1px solid #ccc;padding:2px 6px;">Count2</th>
            <th style="border:1px solid #ccc;padding:2px 6px;">Δ°</th>
            <th style="border:1px solid #ccc;padding:2px 6px;">Δm</th>
        </tr></thead><tbody>`;
        validPairs.forEach((pair, idx) => {
            html += `<tr>
            <td style=\"border:1px solid #ccc;padding:2px 6px;\">${idx + 1}</td>
            <td style=\"border:1px solid #ccc;padding:2px 6px;\">${currentFrameData && currentFrameData.seq_id !== undefined ? currentFrameData.seq_id : '-'}</td>
            <td style=\"border:1px solid #ccc;padding:2px 6px;\">${currentFrameData && currentFrameData.frame_id !== undefined ? currentFrameData.frame_id : '-'}</td>
            <td style=\"border:1px solid #ccc;padding:2px 6px;\">${pair.radar1.replace('RADAR_', '')}</td>
            <td style=\"border:1px solid #ccc;padding:2px 6px;\">${pair.line1Index}</td>
            <td style=\"border:1px solid #ccc;padding:2px 6px;\">${pair.line1Inliers}</td>
            <td style=\"border:1px solid #ccc;padding:2px 6px;\">${pair.radar2.replace('RADAR_', '')}</td>
            <td style=\"border:1px solid #ccc;padding:2px 6px;\">${pair.line2Index}</td>
            <td style=\"border:1px solid #ccc;padding:2px 6px;\">${pair.line2Inliers}</td>
            <td style=\"border:1px solid #ccc;padding:2px 6px;\">${pair.angle.toFixed(2)}</td>
            <td style=\"border:1px solid #ccc;padding:2px 6px;\">${pair.spatialDistance.toFixed(2)}</td>
        </tr>`;
        });
        html += '</tbody></table>';
        analysisDiv.innerHTML = html;

        // CSV copy logic
        const csvBtn = document.getElementById('copyPairsCsvBtn');
        if (csvBtn) {
            csvBtn.onclick = function () {
                const header = [
                    '#', 'seq_id', 'frame_id', 'radar1', 'line1Index', 'line1Inliers',
                    'radar2', 'line2Index', 'line2Inliers', 'angle', 'spatialDistance'
                ];
                const rows = validPairs.map((pair, idx) => [
                    idx + 1,
                    currentFrameData && currentFrameData.seq_id !== undefined ? currentFrameData.seq_id : '',
                    currentFrameData && currentFrameData.frame_id !== undefined ? currentFrameData.frame_id : '',
                    pair.radar1,
                    pair.line1Index,
                    pair.line1Inliers,
                    pair.radar2,
                    pair.line2Index,
                    pair.line2Inliers,
                    pair.angle.toFixed(2),
                    pair.spatialDistance.toFixed(2)
                ]);
                let csv = header.join(',') + '\n' + rows.map(r => r.join(',')).join('\n');
                // Copy to clipboard
                if (navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
                    navigator.clipboard.writeText(csv).then(() => {
                        csvBtn.textContent = 'Copied!';
                        setTimeout(() => { csvBtn.textContent = 'Copy to clipboard as CSV'; }, 1200);
                    }, () => {
                        fallbackCopyTextToClipboard(csv, csvBtn);
                    });
                } else {
                    fallbackCopyTextToClipboard(csv, csvBtn);
                }
            };
        }
        // Fallback function for copying text
        function fallbackCopyTextToClipboard(text, btn) {
            try {
                const textarea = document.createElement('textarea');
                textarea.value = text;
                document.body.appendChild(textarea);
                textarea.select();
                document.execCommand('copy');
                document.body.removeChild(textarea);
                btn.textContent = 'Copied!';
                setTimeout(() => { btn.textContent = 'Copy to clipboard as CSV'; }, 1200);
            } catch (e) {
                btn.textContent = 'Copy failed';
                setTimeout(() => { btn.textContent = 'Copy to clipboard as CSV'; }, 1200);
            }
        }
        // Add event listener for copy button
        const btn = document.getElementById('copyPairsJsonBtn');
        if (btn) {
            btn.onclick = function () {
                // Only copy the main fields, not the THREE.Vector3 objects
                const jsonPairs = validPairs.map(pair => ({
                    seq_id: currentFrameData && currentFrameData.seq_id !== undefined ? currentFrameData.seq_id : null,
                    frame_id: currentFrameData && currentFrameData.frame_id !== undefined ? currentFrameData.frame_id : null,
                    radar1: pair.radar1,
                    radar2: pair.radar2,
                    line1Index: pair.line1Index,
                    line2Index: pair.line2Index,
                    angle: pair.angle,
                    spatialDistance: pair.spatialDistance,
                    line1Inliers: pair.line1Inliers,
                    line2Inliers: pair.line2Inliers
                }));
                const jsonStr = JSON.stringify(jsonPairs, null, 2);
                // Fallback for browsers/environments where navigator.clipboard may be undefined
                if (navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
                    navigator.clipboard.writeText(jsonStr).then(() => {
                        btn.textContent = 'Copied!';
                        setTimeout(() => { btn.textContent = 'Copy as JSON'; }, 1200);
                    }, () => {
                        fallbackCopyTextToClipboard(jsonStr, btn);
                    });
                } else {
                    fallbackCopyTextToClipboard(jsonStr, btn);
                }
            };
        }
        // Fallback function for copying text
        function fallbackCopyTextToClipboard(text, btn) {
            try {
                const textarea = document.createElement('textarea');
                textarea.value = text;
                document.body.appendChild(textarea);
                textarea.select();
                document.execCommand('copy');
                document.body.removeChild(textarea);
                btn.textContent = 'Copied!';
                setTimeout(() => { btn.textContent = 'Copy as JSON'; }, 1200);
            } catch (e) {
                btn.textContent = 'Copy failed';
                setTimeout(() => { btn.textContent = 'Copy as JSON'; }, 1200);
            }
        }
    }
}

// Function to clear all radar data
function clearAllRadarData() {
    radarSensors.forEach(sensor => {
        while (radarGroups[sensor].children.length) {
            radarGroups[sensor].remove(radarGroups[sensor].children[0]);
        }
    });
}
// Function to transform point from sensor to world coordinates
function transformToWorld(point, rotation, translation) {
    // Create rotation matrix from 3x3 array
    const rotMatrix = new THREE.Matrix3();
    rotMatrix.set(
        rotation[0][0], rotation[0][1], rotation[0][2],
        rotation[1][0], rotation[1][1], rotation[1][2],
        rotation[2][0], rotation[2][1], rotation[2][2]
    );

    // Apply rotation
    const rotatedPoint = new THREE.Vector3(point.x, point.y, point.z);
    rotatedPoint.applyMatrix3(rotMatrix);

    // Apply translation
    rotatedPoint.add(new THREE.Vector3(translation[0], translation[1], translation[2]));

    return { x: rotatedPoint.x, y: rotatedPoint.y, z: rotatedPoint.z };
}

// Function to transform all points and lines in data
function transformDataToWorld(data, sensorInfo) {
    if (!sensorInfo || !sensorInfo.rotation || !sensorInfo.translation) {
        return data; // Return original data if no transform info
    }

    const transformedData = {
        points: [],
        lines: []
    };

    // Transform points
    transformedData.points = data.points.map(point =>
        transformToWorld(point, sensorInfo.rotation, sensorInfo.translation)
    );

    // Transform lines
    transformedData.lines = data.lines.map(line => {
        const transformedLine = { ...line };

        // Transform line endpoints
        transformedLine.p0 = transformToWorld(line.p0, sensorInfo.rotation, sensorInfo.translation);
        transformedLine.p1 = transformToWorld(line.p1, sensorInfo.rotation, sensorInfo.translation);
        transformedLine.lineP0 = transformToWorld(line.lineP0, sensorInfo.rotation, sensorInfo.translation);
        transformedLine.lineP1 = transformToWorld(line.lineP1, sensorInfo.rotation, sensorInfo.translation);

        // Transform inlier cloud
        transformedLine.inlierCloud = line.inlierCloud.map(point =>
            transformToWorld(point, sensorInfo.rotation, sensorInfo.translation)
        );

        return transformedLine;
    });

    return transformedData;
}

// Function to get current coordinate system
function getCurrentCoordSystem() {
    return document.querySelector('input[name="coordSystem"]:checked').value;
}

// Function to load and display radar data
function loadRadarData(sensor, filename) {
    fetch(`/json/${filename}`)
        .then(r => r.json())
        .then(data => {
            // Store raw data for coordinate transformations
            currentRadarData[sensor] = data;

            // Clear existing data for this sensor
            while (radarGroups[sensor].children.length) {
                radarGroups[sensor].remove(radarGroups[sensor].children[0]);
            }

            // Transform data based on coordinate system selection
            let displayData = data;
            if (getCurrentCoordSystem() === 'world' && currentFrameData && currentFrameData[sensor]) {
                displayData = transformDataToWorld(data, currentFrameData[sensor]);
            }

            // Add points with sensor-specific color
            const color = radarColors[sensor];
            addPointsToGroup(displayData.points, radarGroups[sensor], color, 0.15);

            // Add lines with sensor-specific color
            const threshold = parseInt(document.getElementById('inlierSlider').value);
            addLinesToGroup(displayData, radarGroups[sensor], color, threshold);

            // Update status
            document.getElementById(`status_${sensor}`).textContent = `✓ ${data.points.length}pts`;
            document.getElementById(`status_${sensor}`).style.color = '#4a9';
        })
        .catch(err => {
            console.error(`Failed to load ${sensor}:`, filename, err);
            document.getElementById(`status_${sensor}`).textContent = '✗ Error';
            document.getElementById(`status_${sensor}`).style.color = '#e44';
        });
}

// Function to refresh all radar data with current coordinate system
function refreshRadarDisplay() {
    radarSensors.forEach(sensor => {
        if (currentRadarData[sensor]) {
            // Clear existing data for this sensor
            while (radarGroups[sensor].children.length) {
                radarGroups[sensor].remove(radarGroups[sensor].children[0]);
            }

            // Transform data based on coordinate system selection
            let displayData = currentRadarData[sensor];
            if (getCurrentCoordSystem() === 'world' && currentFrameData && currentFrameData[sensor]) {
                displayData = transformDataToWorld(currentRadarData[sensor], currentFrameData[sensor]);
            }

            // Add points with sensor-specific color
            const color = radarColors[sensor];
            addPointsToGroup(displayData.points, radarGroups[sensor], color, 0.15);

            // Add lines with sensor-specific color
            const threshold = parseInt(document.getElementById('inlierSlider').value);
            addLinesToGroup(displayData, radarGroups[sensor], color, threshold);
        }
    });
}

// Add event listeners for coordinate system radio buttons
document.querySelectorAll('input[name="coordSystem"]').forEach(radio => {
    radio.addEventListener('change', function () {
        console.log(`Switched to ${this.value} coordinates`);
        refreshRadarDisplay();
        setTimeout(() => analyzeLinePairs(), 100); // Small delay to ensure display is updated
    });
});

// Add event listeners for radar checkboxes
radarSensors.forEach(sensor => {
    const checkbox = document.getElementById(sensor);
    checkbox.addEventListener('change', function () {
        radarGroups[sensor].visible = this.checked;
    });
});

// Add event listener for the show only valid pairs checkbox
document.getElementById('showOnlyValidPairs').addEventListener('change', function () {
    console.log(`Show ONLY valid pairs: ${this.checked}`);
    refreshRadarDisplay(); // Refresh display to apply filtering
    visualizeValidPairs(currentValidPairs); // Update visualization
});
// Update slider to affect all radar sensors
document.getElementById('inlierSlider').addEventListener('input', function () {
    document.getElementById('inlierValue').textContent = this.value;
    const threshold = parseInt(this.value);

    // Refresh display with new threshold
    analyzeLinePairs(); // Analyze pairs when threshold changes
    refreshRadarDisplay();
});

document.getElementById('angleSlider').addEventListener('input', function () {
    document.getElementById('angleValue').textContent = this.value;
    analyzeLinePairs(); // Analyze pairs when threshold changes
    refreshRadarDisplay();
});

document.getElementById('spatialSlider').addEventListener('input', function () {
    document.getElementById('spatialValue').textContent = this.value;
    analyzeLinePairs(); // Analyze pairs when threshold changes
    refreshRadarDisplay();
});


// Function to add points to a specific group
function addPointsToGroup(points, group, color, size = 0.15) {
    let geometry = new THREE.BufferGeometry();
    let positions = [];
    for (let p of points) positions.push(p.x, p.y, p.z);
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    let material = new THREE.PointsMaterial({ color, size });
    let pts = new THREE.Points(geometry, material);
    group.add(pts);
}

// Function to add lines to a specific group
function addLinesToGroup(data, group, color, threshold) {
    const linesGroup = new THREE.Group();
    const inliersGroup = new THREE.Group();
    group.add(linesGroup);
    group.add(inliersGroup);
    const sensorName = group.name; // Get sensor name from group

    for (let [lineIndex, line] of data.lines.entries()) {
        if (line.inliers >= threshold) {
            // Check if we should show this line based on valid pairs filter
            if (isLineInValidPairs(sensorName, lineIndex)) {
                let factor = 30;
                let lineLength = new THREE.Vector3(
                    line.lineP1.x - line.lineP0.x,
                    line.lineP1.y - line.lineP0.y,
                    line.lineP1.z - line.lineP0.z
                ).length();
                let direction = new THREE.Vector3(
                    line.lineP1.x - line.lineP0.x,
                    line.lineP1.y - line.lineP0.y,
                    line.lineP1.z - line.lineP0.z
                ).normalize();
                let extendedP1 = new THREE.Vector3(
                    line.lineP1.x + direction.x * factor * lineLength,
                    line.lineP1.y + direction.y * factor * lineLength,
                    line.lineP1.z + direction.z * factor * lineLength
                );

                // Add extended line (red tint of sensor color)
                addLineToGroup(line.lineP0, extendedP1, linesGroup, lightenColor(color, 0.5), 2);
                // Add inliers (sensor color)
                addPointsToGroup(line.inlierCloud, inliersGroup, color, 0.3);
            }
        }
    }
}

// Function to add a single line to a group
function addLineToGroup(p0, p1, group, color, width = 2) {
    let geometry = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(p0.x, p0.y, p0.z),
        new THREE.Vector3(p1.x, p1.y, p1.z)
    ]);
    let material = new THREE.LineBasicMaterial({ color, linewidth: width });
    let line = new THREE.Line(geometry, material);
    group.add(line);
}

// Function to lighten a color
function lightenColor(color, factor) {
    const r = ((color >> 16) & 0xff) * factor;
    const g = ((color >> 8) & 0xff) * factor;
    const b = (color & 0xff) * factor;
    return (Math.min(255, r) << 16) | (Math.min(255, g) << 8) | Math.min(255, b);
}

// Function to load a specific scene frame
function loadSceneFrame(frameData) {
    currentFrameData = frameData;
    console.log(`Loading Sequence ${frameData.seq_id}, Frame ${frameData.frame_id}`);

    // Clear all existing radar data
    clearAllRadarData();

    // Extract radar file names and load data
    const radarFiles = {};
    radarSensors.forEach(sensor => {
        if (frameData[sensor] && frameData[sensor].image_file) {
            radarFiles[sensor] = frameData[sensor].image_file;
            // Reset status
            document.getElementById(`status_${sensor}`).textContent = 'Loading...';
            document.getElementById(`status_${sensor}`).style.color = '#666';
        } else {
            document.getElementById(`status_${sensor}`).textContent = 'No data';
            document.getElementById(`status_${sensor}`).style.color = '#999';
        }
    });

    // Show cameras if enabled
    if (document.getElementById('showCameras').checked) {
        addAllCameraPlanes(frameData, 300);
    } else {
        clearCameraPlanes();
    }

    console.log('Radar files for this frame:');
    Object.entries(radarFiles).forEach(([sensor, filename]) => {
        console.log(`  ${sensor}: ${filename}`);
        // Load JSON file for this sensor
        const jsonFilename = filename.split('/').pop().replace('.pcd', '.json');
        loadRadarData(sensor, jsonFilename);
    });

    setTimeout(() => analyzeLinePairs(), 500);


    // Listen for camera checkbox changes
    const camCheckbox = document.getElementById('showCameras');
    if (camCheckbox && !camCheckbox._listenerAdded) {
        camCheckbox.addEventListener('change', function () {
            if (this.checked) {
                addAllCameraPlanes(currentFrameData, 300);
            } else {
                clearCameraPlanes();
            }
        });
        camCheckbox._listenerAdded = true;
    }
}

// Add event listeners for radar checkboxes
radarSensors.forEach(sensor => {
    const checkbox = document.getElementById(sensor);
    checkbox.addEventListener('change', function () {
        radarGroups[sensor].visible = this.checked;
    });
});

// Function to load MAN scene data
async function loadSceneData() {
    try {
        const response = await fetch('/man_scene_data.json');
        sceneData = await response.json();
        console.log('Loaded scene data:', sceneData.length, 'frames');
        populateSequenceDropdown();
    } catch (error) {
        console.error('Failed to load scene data:', error);
        document.getElementById('sequenceSelect').innerHTML = '<option value="">Failed to load</option>';
    }
}

// Function to populate the sequence dropdown
function populateSequenceDropdown() {
    const sequenceSelect = document.getElementById('sequenceSelect');
    const frameSelect = document.getElementById('frameSelect');

    if (!sceneData || sceneData.length === 0) {
        sequenceSelect.innerHTML = '<option value="">No data available</option>';
        return;
    }

    // Group by seq_id
    const sequences = {};
    sceneData.forEach(frame => {
        const seqId = frame.seq_id;
        if (!sequences[seqId]) {
            sequences[seqId] = [];
        }
        sequences[seqId].push(frame);
    });

    // Populate sequence dropdown
    sequenceSelect.innerHTML = '<option value="">Select sequence...</option>';
    Object.keys(sequences).sort((a, b) => parseInt(a) - parseInt(b)).forEach(seqId => {
        const option = document.createElement('option');
        option.value = seqId;
        option.textContent = `Sequence ${seqId}`;
        sequenceSelect.appendChild(option);
    });

    // Add event listener for sequence selection
    sequenceSelect.addEventListener('change', function () {
        const selectedSeq = this.value;
        if (selectedSeq) {
            populateFrameDropdown(selectedSeq, sequences[selectedSeq]);
        } else {
            frameSelect.innerHTML = '<option value="">Select sequence first</option>';
        }
    });

    // Auto-select first sequence if available
    const firstSeq = Object.keys(sequences)[0];
    if (firstSeq) {
        sequenceSelect.value = firstSeq;
        populateFrameDropdown(firstSeq, sequences[firstSeq]);
    }
}

// Function to populate the frame dropdown for selected sequence
function populateFrameDropdown(seqId, frames) {
    const frameSelect = document.getElementById('frameSelect');

    // Sort frames by frame_id
    const sortedFrames = frames.sort((a, b) => a.frame_id - b.frame_id);

    // Populate frame dropdown
    frameSelect.innerHTML = '<option value="">Select frame...</option>';
    sortedFrames.forEach(frame => {
        const option = document.createElement('option');
        option.value = frame.frame_id;
        option.textContent = `Frame ${frame.frame_id}`;
        option.frameData = frame; // Store frame data in option
        frameSelect.appendChild(option);
    });

    // Add event listener for frame selection
    frameSelect.removeEventListener('change', handleFrameChange); // Remove old listener
    frameSelect.addEventListener('change', handleFrameChange);

    // Auto-select first frame
    if (sortedFrames.length > 0) {
        frameSelect.value = sortedFrames[0].frame_id;
        loadSceneFrame(sortedFrames[0]);
    }
}

// Handle frame selection change
function handleFrameChange() {
    const frameSelect = document.getElementById('frameSelect');
    const selectedOption = frameSelect.options[frameSelect.selectedIndex];

    if (selectedOption && selectedOption.frameData) {
        loadSceneFrame(selectedOption.frameData);
    }
}




// Initialize file list on page load
// loadFileList();
loadSceneData();



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